import tensorflow as tf
import gpflow
from gpflow.kernels import MultioutputKernel, Combination, Kernel
from gpflow.inducing_variables import FallbackSeparateIndependentInducingVariables


from gpflow import covariances
from gpflow.config import default_float, default_jitter
from gpflow.inducing_variables import (
    FallbackSeparateIndependentInducingVariables,
    FallbackSharedIndependentInducingVariables,
    InducingPoints,
    SeparateIndependentInducingVariables,
    SharedIndependentInducingVariables,
)
from gpflow.kernels import (
    Combination,
    IndependentLatent,
    LinearCoregionalization,
    MultioutputKernel,
    SeparateIndependent,
    SharedIndependent,
)
from gpflow.conditionals.dispatch import conditional
from gpflow.conditionals.util import (
    base_conditional,
    expand_independent_outputs,
    fully_correlated_conditional,
    independent_interdomain_conditional,
    mix_latent_gp,
    rollaxis_left,
)

class PartlySharedIndependent(SeparateIndependent):
    """
    - Shared: we use a different kernel for each input dimension,
            but all latent GP's that correspond to this input dimension share the same hyperparameter.
            I.e. for D input and DxP output dims, use D independent kernels.
    - Independent: Latents are uncorrelated a priori.
    """

    def __init__(self, kernels, name=None, DoF=None):
        super().__init__(kernels=kernels, name=name)
        self.D = len(kernels)
        self.DoF = self.D+1 if DoF is not None else DoF

    def K(self, X, X2=None, full_output_cov=True):
        if full_output_cov:
            Kxxs = []
            for k in self.kernels:
                Kxxs.append(k.K(X,X2))
            tf.print(len(Kxxs))
            Kxxs = tf.stack(Kxxs, axis=2)
            tf.print(Kxxs.shape)
            Kxxs = tf.stack([k.K(X, X2) for k in self.kernels], axis=2)  # [N, N2, P]
            return tf.transpose(tf.linalg.diag(Kxxs), [0, 2, 1, 3])  # [N, P, N2, P]
        else:
            return tf.stack([k.K(X, X2) for k in self.kernels], axis=0)  # [P, N, N2]


@conditional.register(object, SharedIndependentInducingVariables, PartlySharedIndependent, object)
def partly_shared_independent_conditional(
    Xnew,
    inducing_variable,
    kernel,
    f,
    *,
    full_cov=False,
    full_output_cov=False,
    q_sqrt=None,
    white=False,
):
    """Multi-output GP with independent GP priors.
    Number of latent processes equals the number of outputs (L = P).
    The covariance matrices used to calculate the conditional have the following shape:
    - Kuu: [P, M, M]
    - Kuf: [P, M, N]
    - Kff: [P, N] or [P, N, N]

    Further reference
    -----------------
    - See `gpflow.conditionals._conditional` for a detailed explanation of
      conditional in the single-output case.
    - See the multioutput notebook for more information about the multioutput framework.
    - See above for the parameters and the return value.
    """
    # Following are: [P, M, M]  -  [P, M, N]  -  [P, N](x N)
    tf.print('hello')
    Kmms = covariances.Kuu(inducing_variable, kernel, jitter=default_jitter())  # [P, M, M]
    Kmns = covariances.Kuf(inducing_variable, kernel, Xnew)  # [P, M, N]
    if isinstance(kernel, Combination):
        kernels = kernel.kernels
    else:
        kernels = [kernel.kernel] * len(inducing_variable.inducing_variable_list)
    Knns = tf.stack([k.K(Xnew) if full_cov else k.K_diag(Xnew) for k in kernels], axis=0)
    fs = tf.transpose(f)[:, :, None]  # [P, M, 1]
    # [P, 1, M, M]  or  [P, M, 1]

    if q_sqrt is not None:
        q_sqrts = (
            tf.transpose(q_sqrt)[:, :, None] if q_sqrt.shape.ndims == 2 else q_sqrt[:, None, :, :]
        )
        base_conditional_args_to_map = (Kmms, Kmns, Knns, fs, q_sqrts)

        def single_gp_conditional(t):
            Kmm, Kmn, Knn, f, q_sqrt = t
            return base_conditional(Kmn, Kmm, Knn, f, full_cov=full_cov, q_sqrt=q_sqrt, white=white)

    else:
        base_conditional_args_to_map = (Kmms, Kmns, Knns, fs)

        def single_gp_conditional(t):
            Kmm, Kmn, Knn, f = t
            return base_conditional(Kmn, Kmm, Knn, f, full_cov=full_cov, q_sqrt=q_sqrt, white=white)

    rmu, rvar = tf.map_fn(
        single_gp_conditional, base_conditional_args_to_map, (default_float(), default_float())
    )  # [P, N, 1], [P, 1, N, N] or [P, N, 1]

    fmu = rollaxis_left(tf.squeeze(rmu, axis=-1), 1)  # [N, P]

    if full_cov:
        fvar = tf.squeeze(rvar, axis=-3)  # [..., 0, :, :]  # [P, N, N]
    else:
        fvar = rollaxis_left(tf.squeeze(rvar, axis=-1), 1)  # [N, P]

    return fmu, expand_independent_outputs(fvar, full_cov, full_output_cov)