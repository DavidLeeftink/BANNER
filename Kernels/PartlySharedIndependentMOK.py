import tensorflow as tf
import gpflow
from gpflow.kernels import MultioutputKernel, Combination, Kernel
from gpflow.inducing_variables import FallbackSeparateIndependentInducingVariables
from gpflow.covariances.dispatch import Kuf, Kuu
from typing import Union
from gpflow import covariances
from gpflow.config import default_float, default_jitter
from gpflow.base import TensorLike
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

class CustomSeparateIndependent(MultioutputKernel, Combination):
    """
    - Separate: we use different kernel for each output latent
    - Independent: Latents are uncorrelated a priori.
    """

    def __init__(self, kernels, name=None):
        super().__init__(kernels=kernels, name=name)
        print('made it to init')

    @property
    def num_latent_gps(self):
        return len(self.kernels)

    @property
    def latent_kernels(self):
        """The underlying kernels in the multioutput kernel"""
        return tuple(self.kernels)

    def K(self, X, X2=None, full_output_cov=True):
        tf.print('made it to K()')
        if full_output_cov:
            Kxxs = tf.stack([k.K(X, X2) for k in self.kernels], axis=2)  # [N, N2, P]
            return tf.transpose(tf.linalg.diag(Kxxs), [0, 2, 1, 3])  # [N, P, N2, P]
        else:
            return tf.stack([k.K(X, X2) for k in self.kernels], axis=0)  # [P, N, N2]

    def K_diag(self, X, full_output_cov=False):
        tf.print('made it to K_diag')
        stacked = tf.stack([k.K_diag(X) for k in self.kernels], axis=1)  # [N, P]
        return tf.linalg.diag(stacked) if full_output_cov else stacked  # [N, P, P]  or  [N, P]


@Kuf.register(SharedIndependentInducingVariables, CustomSeparateIndependent, object)
def _Kuf(
    inducing_variable: SeparateIndependentInducingVariables,
    kernel: CustomSeparateIndependent,
    Xnew: tf.Tensor,
):
    return tf.stack(
        [Kuf(f, kernel.kernel, Xnew) for f in inducing_variable.inducing_variable_list], axis=0
    )  # [L, M, N]

@Kuu.register(SharedIndependentInducingVariables, CustomSeparateIndependent)
def _Kuu(
    inducing_variable: FallbackSharedIndependentInducingVariables,
    kernel: Union[SeparateIndependent, IndependentLatent],
    *,
    jitter=0.0,
):
    Kmm = tf.stack(
        [Kuu(inducing_variable.inducing_variable, k) for k in kernel.kernels], axis=0
    )  # [L, M, M]
    jittermat = tf.eye(inducing_variable.num_inducing, dtype=Kmm.dtype)[None, :, :] * jitter
    return Kmm + jittermat

@conditional.register(object, SharedIndependentInducingVariables, CustomSeparateIndependent, object)
@conditional.register(object, SeparateIndependentInducingVariables, CustomSeparateIndependent, object)
def custom_separate_independent_conditional(
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
    tf.print('hgello')
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