import numpy as np
import gpflow
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from gpflow.likelihoods.base import ScalarLikelihood


class WishartLikelihoodBase(ScalarLikelihood):
    """
    Abstract class for all Wishart Processes likelihoods.
    """
    def __init__(self, D, DoF, R=10, model_inverse=True, additive_noise=True, **kwargs):
        # Todo: confirm if latent dim is correctly specified. gpflow 1 did not require this
        # Note: gpflow transforms -> replaced by tensorflow_probability.bijectors
        """
        :param D (int) Covariance matrix dimension
        :param DoF (int) Degrees of freedom
        :param R (int) Number of monte carlo samples used to approximate reparameterized gradients.
        :param inverse (bool) Use the inverse Wishart Process if true, otherwise use the standard Wishart Process.
        :param additive_noise (bool) If true, the additive white noise model likelihood is used for more robustness in the Wishart Process model.
        """
        super().__init__()#latent_dim=D*DoF, observation_dim=D)
        self.D = D
        self.DoF = DoF
        self.R = R
        self.model_inverse = model_inverse
        self.additive_noise = additive_noise

    def _variational_expectations(self, f_mean, f_cov, Y):
        """
        Function written by Creighton Heaukulani and Mark van der Wilk, adapted for gpflow 2

        Calculate log p(Y | variational parameters)

        :param f_mean: (N, D*DoF), mean parameters of latent GP points F
        :param f_cov: (N, D*DoF), covariance parameters of latent GP points F
        :param Y: (N, D), observations
        :return logp: (N,), log probability density of the data.
        """

        N, D = tf.shape(Y)
        DoF = f_mean.shape[1]/D

        ## Produce R samples of F (latent GP points at the input locations X). TF automatically differentiates through this.
        W = tf.random.normal([self.R, N, D*DoF])
        f_sample = W * (f_cov ** 0.5) + f_mean
        f_sample = tf.reshape(f_sample, [self.R, N, D, -1])

        ## compute the (mean of the) likelihood
        logp = self._log_prob(f_sample, Y)
        return tf.reduce_mean(logp, axis=0)

    def _scalar_log_prob(self, F, Y):
        """
        Log probability of covariance matrix Sigma_n = A F_n F_n^T A^T
        Implements equation (5) in Heaukulani-van der Wilk
        :param

        """
        N, D = tf.shape(Y)
        D = tf.dtypes.cast(D, tf.float32)
        log_det_cov, yt_inv_y = self.make_gaussian_components(F,Y) # (R, N), (R,N)
        log_p = - 0.5*D* np.log(2*np.pi) - 0.5*log_det_cov - 0.5*yt_inv_y # (R,N)
        return tf.reduce_mean(log_p, axis=0) # (N,)

        # ## This goes to Gaussian Components class.
        # AF = self.A_diag[:, None] * tf.reshape(F, [self.R, N, D, -1])
        # yffy = tf.reduce_sum(tf.einsum('jk, ijkl− > ijl', Y, AF) ** 2.0, axis=-1)
        # chols = tf.linalg.cholesky(tf.matmul(AF, AF, transpose_b=True))  # cholesky of precision
        # log_chols = tf.math.log(tf.linalg.diag_part(chols)) - 0.5 * yffy

        ## End gaussian components class
        # Note that maybe _logp() should be overwritten. This depends if the reduce_sum over all data points is case dependent (then yes) or always required (then no)
        #log_p =
        #return log_chols


    def _conditional_mean(self, F):
        raise NotImplementedError

    def _conditional_variance(self, F):
        raise NotImplementedError

    def make_gaussian_components(self, F, Y):
        """
        Returns components used in the Gaussian density kernels
        Abstract method, should be implemented by concrete classes.
        :param F (R, N, D, ),  the (samples) matrix of GP outputs.
        :param Y (N,D) observations
        """
        raise NotImplementedError

class FullCovLikelihood(WishartLikelihoodBase):
    """
    Concrete class for the full covariance likelihood models.
    """
    def __init__(self, D, DoF, **kwargs):
        assert DoF>=D, "Degrees of freedom must be larger or equal than the number of dimensionality of the covariance matrix"
        super().__init__(D, DoF, **kwargs)
