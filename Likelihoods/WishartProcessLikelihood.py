import numpy as np
import gpflow
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from gpflow.likelihoods.base import ScalarLikelihood


class WishartProcessLikelihoodBase(ScalarLikelihood):
    """
    Abstract class for all Wishart Processes likelihoods.
    """
    def __init__(self, D, DoF, R=10, model_inverse=True, **kwargs):
        # Todo: confirm if latent dim is correctly specified. gpflow 1 did not require this
        # Note: gpflow transforms -> replaced by tensorflow_probability.bijectors
        """
        :param D (int) Covariance matrix dimension
        :param DoF (int) Degrees of freedom
        :param R (int) Number of monte carlo samples used to approximate reparameterized gradients.
        :param inverse (bool) Use the inverse Wishart Process if true, otherwise use the standard Wishart Process.
        """
        super().__init__()#latent_dim=D*DoF, observation_dim=D)
        self.D = D
        self.DoF = DoF
        self.R = R
        self.model_inverse = model_inverse

    def _variational_expectations(self, f_mean, f_cov, Y):
        # ToDo: should the name be with '_' in front?
        """
        Function written by Creighton Heaukulani and Mark van der Wilk, adapted for tensorflow 2.4

        Calculate log p(Y | variational parameters)

        :param f_mean: (N, D*DoF), mean parameters of latent GP points F
        :param f_cov: (N, D*DoF), covariance parameters of latent GP points F
        :param Y: (N, D), observations
        :return logp: (N,), log probability density of the data.
        """

        N, D = tf.shape(Y)
        DoF = f_mean.shape[1]/D

        # Produce R samples of F (latent GP points at the input locations X). TF automatically differentiates through this.
        W = tf.random.normal([self.R, N, D*DoF])
        F = W * (f_cov ** 0.5) + f_mean
        F = tf.reshape(F, [self.R, N, D, -1])

        # compute the (mean of the) likelihood


        ## start scalar log prob code
        AF = self.A_diag[:, None] * tf.reshape(F, [self.R, N, D, -1])
        yffy = tf.reduce_sum(tf.einsum('jk, ijkl− > ijl', Y, AF)** 2.0, axis=-1)
        chols = tf.linalg.cholesky(tf.matmul(AF, AF, transpose_b=True))
        log_chols = tf.math.log(tf.linalg.diag_part(chols))
        logp = self._tf.reduce_sum(), axis=2)
        ## end scalar log prob code

        logp = self._log_prob(F, Y) - 0.5 * yffy
        return tf.reduce_mean(logp, axis=0)

    def _scalar_log_prob(self, F, Y):
        """
        Log probability of
        :param

        """
        N, D = tf.shape(Y)
        AF = self.A_diag[:, None] * tf.reshape(F, [self.R, N, D, -1])
        yffy = tf.reduce_sum(tf.einsum('jk, ijkl− > ijl', Y, AF) ** 2.0, axis=-1)
        chols = tf.linalg.cholesky(tf.matmul(AF, AF, transpose_b=True))  # cholesky of precision logp = tf.reduce_sum(tf.log(tf.matrix_diag_part(chols)), axis=2) − 0.5 ∗ yffy return tf .reduce_mean(logp , axis=0)
        log_chols = tf.math.log(tf.linalg.diag_part(chols))

        return log_chols

    def _log_prob(self, F, Y):


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