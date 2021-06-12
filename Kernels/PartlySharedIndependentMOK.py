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
import time


class CustomMultiOutput(MultioutputKernel, Combination):
    """
    - Separate: we use different kernel for each output latent
    - Independent: Latents are uncorrelated a priori.
    """

    def __init__(self, kernels, nu, name=None):
        #kernels = [SharedIndependent(k, output_dim=nu) for k in kernels]
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
        # todo: these are not yet tested with new for loops for partially shared
        if full_output_cov:
            Kxxs = [k.K(X, X2) for k in self.kernels]
            Kxxs = tf.tile(Kxxs, multiples=[self.nu, 1,1])
            Kxxs = tf.stack(Kxxs, axis=2)  # [N, N2, P] # todo: possibly redundant line of code
            return tf.transpose(tf.linalg.diag(Kxxs), [0, 2, 1, 3])  # [N, P, N2, P]
        else:
            return tf.stack([k.K(X, X2) for k in self.kernels], axis=0)  # [P, N, N2]

    def K_diag(self, X, full_output_cov=False):
        # todo: not yet tested with .tile for copies of kernel
        k_diags = [k.K_diag(X) for k in self.kernels]
        k_diags = tf.tile(k_diags, multiples=[self.nu, 1])
        stacked = tf.stack(k_diags, axis=1)  # [N, P] # todo: possibly redundant to do this.
        return tf.linalg.diag(stacked) if full_output_cov else stacked  # [N, P, P]  or  [N, P]


@Kuf.register(SharedIndependentInducingVariables, CustomMultiOutput, object)
def _Kuf( inducing_variable: SharedIndependentInducingVariables,
          kernel: CustomMultiOutput, Xnew: tf.Tensor):
    M = inducing_variable.num_inducing
    N = Xnew.shape[0]
    # slow but working solution working with copies:
    Kmfs = tf.stack([make_copies(Kuf(inducing_variable.inducing_variable, k, Xnew), kernel.nu)
                     for k in kernel.kernels], axis=0) # [D, nu, M, N] # new line
    return tf.reshape(Kmfs, (-1, M, N)) # [L, M, N] # new line

@Kuu.register(SharedIndependentInducingVariables, CustomMultiOutput)
def _Kuu( inducing_variable: SharedIndependentInducingVariables,
          kernel: Union[SeparateIndependent, IndependentLatent], *, jitter=0.0):
    M = inducing_variable.num_inducing

    ## working solution with copies (slow)
    Kmms = tf.stack([make_copies(Kuu(inducing_variable.inducing_variable, k), kernel.nu)
                   for k in kernel.kernels], axis=0)  # [D, M, M] # new line
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
    Kmms = covariances.Kuu(inducing_variable, kernel, jitter=default_jitter())  # [P, M, M]
    Kmns = covariances.Kuf(inducing_variable, kernel, Xnew)  # [P, M, N]
    if isinstance(kernel, Combination):
        kernels = kernel.kernels
    else:
        kernels = [kernel.kernel] * len(inducing_variable.inducing_variable_list)
    M = inducing_variable.num_inducing
    N = Xnew.shape[0]
    time1_start = time.time()

    ## working (but slow) copying solution:
    Knns = [make_copies(k.K(Xnew) if full_cov else k.K_diag(Xnew), kernel.nu, n_dims=len(Xnew.shape)-1+1*full_cov) for k in kernels]
    Knns = tf.reshape(tf.stack(Knns,axis=0), (-1, N)) if not full_cov else tf.reshape(Knns, (-1, N, N))

    fs = tf.transpose(f)[:, :, None]  # [P, M, 1]

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
    print('before and after for rmu and fmu: ', rmu.shape, fmu.shape)
    if full_cov:
        fvar = tf.squeeze(rvar, axis=-3)  # [..., 0, :, :]  # [P, N, N]
    else:
        fvar = rollaxis_left(tf.squeeze(rvar, axis=-1), 1)  # [N, P]
    print('ful cov: ', full_cov)
    print('fmeans post transpose: ', fmu.shape)
    print('fvars post transpose: ', fvar.shape)
    return fmu, expand_independent_outputs(fvar, full_cov, full_output_cov)

@tf.function
def make_copies(matrix, N, n_dims=2):
    """
    :param tf matrix of (NxM)
    """
    if n_dims==2:
        return tf.tile(matrix, multiples=[N,1])
    else:
        return tf.tile(matrix, multiples=[N])


class CustomMultiOutput2(MultioutputKernel, Combination):
    """
    - Separate: we use different kernel for each output latent
    - Independent: Latents are uncorrelated a priori.
    """

    def __init__(self, kernels, nu, name=None):
        kernels = [SharedIndependent(k, output_dim=nu) for k in kernels]
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
        # todo: these are not yet tested with new for loops for partially shared
        if full_output_cov:
            Kxxs = [k.K(X, X2) for k in self.kernels]
            Kxxs = tf.tile(Kxxs, multiples=[self.nu, 1,1])
            Kxxs = tf.stack(Kxxs, axis=2)  # [N, N2, P] # todo: possibly redundant line of code
            return tf.transpose(tf.linalg.diag(Kxxs), [0, 2, 1, 3])  # [N, P, N2, P]
        else:
            return tf.stack([k.K(X, X2) for k in self.kernels], axis=0)  # [P, N, N2]

    def K_diag(self, X, full_output_cov=False):
        # todo: not yet tested with .tile for copies of kernel
        k_diags = [k.K_diag(X) for k in self.kernels]
        k_diags = tf.tile(k_diags, multiples=[self.nu, 1])
        stacked = tf.stack(k_diags, axis=1)  # [N, P] # todo: possibly redundant to do this.
        return tf.linalg.diag(stacked) if full_output_cov else stacked  # [N, P, P]  or  [N, P]
@conditional.register(object, SharedIndependentInducingVariables, CustomMultiOutput2, object)
def custom_shared_independent_conditional(
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
    """Multioutput conditional for an independent kernel and shared inducing inducing.
    Same behaviour as conditional with non-multioutput kernels.
    The covariance matrices used to calculate the conditional have the following shape:
    - Kuu: [M, M]
    - Kuf: [M, N]
    - Kff: N or [N, N]

    Further reference
    -----------------
    - See `gpflow.conditionals._conditional` for a detailed explanation of
      conditional in the single-output case.
    - See the multioutput notebook for more information about the multioutput framework.
    Parameters
    ----------
    :param Xnew: data matrix, size [N, D].
    :param f: data matrix, [M, P]
    :param full_cov: return the covariance between the datapoints
    :param full_output_cov: return the covariance between the outputs.
        Note: as we are using a independent kernel these covariances will be zero.
    :param q_sqrt: matrix of standard-deviations or Cholesky matrices,
        size [M, P] or [P, M, M].
    :param white: boolean of whether to use the whitened representation
    :return:
        - mean:     [N, P]
        - variance: [N, P], [P, N, N], [N, P, P] or [N, P, N, P]
        Please see `gpflow.conditional._expand_independent_outputs` for more information
        about the shape of the variance, depending on `full_cov` and `full_output_cov`.
    """
    N, P = Xnew.shape[0], q_sqrt.shape[0]
    nu = kernel.nu
    fmeans, fvars = [], []

    for i, k in enumerate(kernel.kernels):
        nu_idx_start, nu_idx_end = int(i * kernel.nu), int((i + 1) * kernel.nu)
        f_i = f[:, nu_idx_start:nu_idx_end]
        q_sqrt_i = q_sqrt[nu_idx_start:nu_idx_end, :]
        Kmm = covariances.Kuu(inducing_variable, k, jitter=default_jitter())  # [M, M]
        Kmn = covariances.Kuf(inducing_variable, k, Xnew)  # [M, N]
        Knn = k.kernel(Xnew, full_cov=full_cov)

        fmean, fvar = base_conditional(
            Kmn, Kmm, Knn, f_i, full_cov=full_cov, q_sqrt=q_sqrt_i, white=white
        )  # [N, P],  [P, N, N] or [N, P]
        fmeans.append(tf.transpose(fmean,perm=[1,0]))
        if full_cov:
            fvars.append(tf.transpose(fvar,perm=[0,2,1]))
        else:
            fvars.append(tf.transpose(fvar,perm=[1,0]))
        #fvars_perm = [1,0] if len(fvar.shape)==2 else [0,2,1]
        #fvars.append(tf.transpose(fvar,perm=fvars_perm)) # transpose needs further perm for gwp to work!

    fmeans = tf.transpose(tf.reshape(fmeans,shape=(-1,N)))
    if full_cov:
        fvars = tf.reshape(fvars,shape=(-1,N,N))
    else:
        fvars = tf.transpose(tf.reshape(fvars, shape=(-1,N)))

    print('fmeans post transpose: ', fmeans.shape)
    print('fvars post transpose: ',fvars.shape)
    return fmeans, expand_independent_outputs(fvars, full_cov, full_output_cov)