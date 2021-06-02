import itertools
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
import numpy as np

class CustomMultiOutput(MultioutputKernel, Combination):
    """
    - Separate: we use different kernel for each output latent
    - Independent: Latents are uncorrelated a priori.
    """

    def __init__(self, kernels, nu, name=None):
        super().__init__(kernels=kernels, name=name)
        self.nu = nu

    @property
    def num_latent_gps(self):
        return len(self.kernels)

    @property
    def latent_kernels(self):
        """The underlying kernels in the multioutput kernel"""
        return tuple(self.kernels)

    def K(self, X, X2=None, full_output_cov=True):
        tf.print('made it to K()')
        # todo: these are not yet updated with new for loops for partially shared
        # todo: location 1a with for-loop over kernels
        # q: why is each kernel applied to the same X and X2 matrices?
        if full_output_cov:
            Kxxs = [k.K(X, X2) for k in self.kernels]
            # for each kernel k.K(), make self.nu copies. a
            Kxxs = tf.stack(Kxxs, axis=2)  # [N, N2, P]
            return tf.transpose(tf.linalg.diag(Kxxs), [0, 2, 1, 3])  # [N, P, N2, P]
        else:
            return tf.stack([k.K(X, X2) for k in self.kernels], axis=0)  # [P, N, N2]

    def K_diag(self, X, full_output_cov=False):
        tf.print('made it to K_diag')
        # todo: location 1b with for loop over kernels
        k_diags = [k.K_diag(X) for k in self.kernels]
        stacked = tf.stack(k_diags, axis=1)  # [N, P]
        return tf.linalg.diag(stacked) if full_output_cov else stacked  # [N, P, P]  or  [N, P]


@Kuf.register(SharedIndependentInducingVariables, CustomMultiOutput, object)
def _Kuf( inducing_variable: SharedIndependentInducingVariables,
          kernel: CustomMultiOutput, Xnew: tf.Tensor):
    # todo: location 2 where changes are required.
    M = inducing_variable.num_inducing
    N = Xnew.shape[0]
    Kmfs = tf.stack([make_copies(Kuf(inducing_variable.inducing_variable, k, Xnew), kernel.nu)
                     for k in kernel.kernels], axis=0) # [D, nu, M, N] # new line
    return tf.reshape(Kmfs, (-1, M, N)) # [L, M, M] # new line
    # return tf.stack([Kuf(inducing_variable.inducing_variable, k, Xnew) for k in kernel.kernels], axis=0)  # [L, M, N] # old line

@Kuu.register(SharedIndependentInducingVariables, CustomMultiOutput)
def _Kuu( inducing_variable: SharedIndependentInducingVariables,
          kernel: Union[SeparateIndependent, IndependentLatent], *, jitter=0.0):
    # todo: location 2b where changes are required
    #Kmm = tf.stack([Kuu(inducing_variable.inducing_variable, k) for k in kernel.kernels], axis=0 )  # [L, M, M] # old line
    M = inducing_variable.num_inducing
    Kmms = tf.stack([make_copies(Kuu(inducing_variable.inducing_variable, k),kernel.nu)
                     for k in kernel.kernels], axis=0) # [D, M, M] # new line
    Kmms = tf.reshape(Kmms, shape=(-1, M, M))
    jittermat = tf.eye(inducing_variable.num_inducing, dtype=Kmms.dtype)[None, :, :] * jitter
    return Kmms + jittermat

@conditional.register(object, SharedIndependentInducingVariables, CustomMultiOutput, object)
@conditional.register(object, SeparateIndependentInducingVariables, CustomMultiOutput, object)
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
    M = inducing_variable.num_inducing
    N = Xnew.shape[0]
    Kmms = covariances.Kuu(inducing_variable, kernel, jitter=default_jitter())  # [P, M, M]
    Kmns = covariances.Kuf(inducing_variable, kernel, Xnew)  # [P, M, N]
    if isinstance(kernel, Combination):
        kernels = kernel.kernels
    else:
        kernels = [kernel.kernel] * len(inducing_variable.inducing_variable_list)
    # todo: location 3 where change is required.

    #Knns = [k.K(Xnew) if full_cov else k.K_diag(Xnew) for k in kernels] # original line:
    Knns = [make_copies(k.K(Xnew) if full_cov else k.K_diag(Xnew), kernel.nu, n_dims=len(Xnew.shape)-1+1*full_cov) for k in kernels]
    Knns = tf.reshape(tf.stack(Knns,axis=0), (-1, N)) if not full_cov else tf.reshape(Knns, (-1, N, N))
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

@tf.function
def make_copies(matrix, N, n_dims=2):
    """
    :param tf matrix of (NxM)
    """
    if n_dims==2:
        return tf.tile(tf.expand_dims(matrix, axis=0), multiples=[N,1,1])
    else:
        return tf.tile(tf.expand_dims(matrix, axis=0), multiples=[N,1])