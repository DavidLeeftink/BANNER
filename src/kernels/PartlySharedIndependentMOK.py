import tensorflow as tf
import gpflow
from gpflow.kernels import MultioutputKernel, Combination, Kernel
from gpflow import covariances
from gpflow.config import default_float, default_jitter
from gpflow.base import TensorLike
from gpflow.inducing_variables import (
    SharedIndependentInducingVariables,
)
from gpflow.kernels import (
    Combination,
    MultioutputKernel,
    SeparateIndependent,
    SharedIndependent,
)
from gpflow.conditionals.dispatch import conditional
from gpflow.conditionals.util import (
    base_conditional,
    expand_independent_outputs
)


class PartlySharedIndependentMultiOutput(MultioutputKernel, Combination):
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

@conditional.register(object, SharedIndependentInducingVariables, PartlySharedIndependentMultiOutput, object)
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

    fmeans = tf.transpose(tf.reshape(fmeans,shape=(-1,N)))
    if full_cov:
        fvars = tf.reshape(fvars,shape=(-1,N,N))
    else:
        fvars = tf.transpose(tf.reshape(fvars, shape=(-1,N)))

    return fmeans, expand_independent_outputs(fvars, full_cov, full_output_cov)