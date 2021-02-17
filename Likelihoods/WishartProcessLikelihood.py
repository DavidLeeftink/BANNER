import numpy as np
import gpflow
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from gpflow.likelihoods.base import ScalarLikelihood


class InvWishartProcessLikelihood(ScalarLikelihood):
    def __init__(self, D, DoF, R=1):
        # To do: confirm if latent dim is correctly specified. gpflow 1 did not require this
        super().__init__(latent_dim=D*DoF, observation_dim=D)
        self.D = D
        self.DoF = DoF
        self.R = R # n_samples for variational expectations for automatic differentiation

        # Note: gpflow transforms -> replaced by tensorflow_probability.bijectors

    def _variational_expectations(self, mu, S, Y):
        # ToDo: should the name be with '_' in front?
        """
        Function written by Creighton Heaukulani and Mark van der Wilk, adapted for tensorflow 2.4
        """
        N, D = tf.shape(Y)
        W = tf.random.normal([self.R, N, tf.shape(mu)[1]])
        F = W*(S**0.5) + mu  # samples through which TF automatically differentiates
        # compute the (mean of the) likelihood
        AF = self.A_diag[:, None] * tf.reshape(F, [self.R, N, D, -1])
        yffy = tf.reduce_sum(tf.einsum('jk, ijkl− > ijl', Y, AF)** 2.0, axis=-1)
        chols = tf.linalg.cholesky(tf.matmul(AF, AF, transpose_b=True))  # cholesky of precision logp = tf.reduce_sum(tf.log(tf.matrix_diag_part(chols)), axis=2) − 0.5 ∗ yffy return tf .reduce_mean(logp , axis=0)
        logp = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chols)), axis=2) - 0.5 * yffy
        return tf.reduce_mean(logp, axis=0)