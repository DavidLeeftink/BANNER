import numpy as np
import gpflow
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.likelihoods.base import ScalarLikelihood
from gpflow.utilities import positive
from gpflow import Parameter


class WishartLikelihoodBase(ScalarLikelihood):
    """
    Abstract class for all Wishart Processes likelihoods.
    Class written by Creighton Heaukulani and Mark van der Wilk, and is adapted for gpflow 2.
    """
    def __init__(self, D, DoF, R=10, model_inverse=True, additive_noise=True, **kwargs):
        """
        :param D (int) Covariance matrix dimension
        :param DoF (int) Degrees of freedom
        :param R (int) Number of monte carlo samples used to approximate reparameterized gradients.
        :param inverse (bool) Use inverse Wishart Process if true, otherwise standard Wishart Process.
        :param additive_noise (bool) Use additive white noise model likelihood if true.
        """
        super().__init__()  # todo: check likelihoods' specification of dimensions
        self.D = D
        self.DoF = DoF
        self.R = R
        self.model_inverse = model_inverse
        self.additive_noise = additive_noise

    def variational_expectations(self, f_mean, f_cov, Y):
        """
        Calculate log p(Y | variational parameters)

        :param f_mean: (N, D*DoF), mean parameters of latent GP points F
        :param f_cov: (N, D*DoF), covariance parameters of latent GP points F
        :param Y: (N, D), observations
        :return logp: (N,), log probability density of the data.
        where N is the minibatch size, D the covariance matrix dimension and DoF the degrees of freedom
        """
        _, latent_dim = f_mean.shape
        _, D = Y.shape
        N = tf.shape(Y)[0]

        # Produce R samples of F (latent GP points at the input locations X).
        # TF automatically differentiates through this.
        W = tf.dtypes.cast(tf.random.normal(shape=[self.R, N, latent_dim]), tf.float64)
        f_sample = W * f_cov**0.5 + f_mean
        f_sample = tf.reshape(f_sample, [self.R, N, D, -1])
        # compute the mean of the likelihood
        logp = self._log_prob(f_sample, Y) #(N,)
        return logp

    def _log_prob(self, F, Y): # (R,N) -> (N)
        return tf.math.reduce_mean(self._scalar_log_prob(F, Y), axis=0)

    def _scalar_log_prob(self, F, Y):
        """
        Log probability of covariance matrix Sigma_n = A F_n F_n^T A^T
        Implements equation (5) in Heaukulani-van der Wilk
        :param F (R,N,D) the (sampled) matrix of GP outputs
        :param Y (N,D) observations
        """
        D = tf.dtypes.cast(self.D, tf.float64)
        log_det_cov, yt_inv_y = self.make_gaussian_components(F,Y) # (R, N), (R,N)
        log_p = - 0.5 * D * np.log(2*np.pi) - 0.5*log_det_cov - 0.5*yt_inv_y # (R,N)
        return log_p # (R,N)

    def make_gaussian_components(self, F, Y):
        """
        Returns components used in the Gaussian density kernels
        Abstract method, should be implemented by concrete classes.
        :param F (R, N, D, ),  the (sampled) matrix of GP outputs.
        :param Y (N,D) observations
        """
        raise NotImplementedError


class FullWishartLikelihood(WishartLikelihoodBase):
    """
    Concrete class for the full covariance likelihood models.
    The code is written by Heaukulani-van der Wilk (see references above)
    """
    def __init__(self, D, DoF, A=None, **kwargs):
        """
        :param D (int) Dimensionality of covariance matrix
        :param DoF (int) degrees of freedom
        :param A (DxD matrix) scale matrix. Default is a DxD identity matrix.
        """
        assert DoF>=D, "Degrees of freedom must be larger or equal than the dimensionality of the covariance matrix"
        super().__init__(D, DoF, **kwargs)

        # this case assumes a square scale matrix, and it must lead with dimension D
        self.A = A if A is not None else Parameter(np.ones(self.D), transform=positive(), dtype=tf.float64)
        gpflow.set_trainable(self.A, False)

        if self.additive_noise:
            # create additional noise param; should be positive; conc=0.1 and rate=0.0001 initializes sigma2inv=1000 and thus initializes sigma2=0.001
            self.p_sigma2inv_conc = Parameter(0.1, transform=positive(), dtype=tf.float64)
            self.p_sigma2inv_rate = Parameter(0.0001, transform=positive(), dtype=tf.float64)
            self.q_sigma2inv_conc = Parameter(0.1 * np.ones(self.D), transform=positive(), dtype=tf.float64)
            self.q_sigma2inv_rate = Parameter(0.0001 * np.ones(self.D), transform=positive(), dtype=tf.float64)
            # Todo: double check if these should be trainable
            # gpflow.set_trainable(self.p_sigma2inv_conc, False)
            # gpflow.set_trainable(self.p_sigma2inv_rate, False)
            # gpflow.set_trainable(self.q_sigma2inv_conc, False)
            # gpflow.set_trainable(self.q_sigma2inv_rate, False)


    def make_gaussian_components(self, F, Y):
        """
        An auxiliary function for logp that returns the complexity pentalty and the data fit term.
        Note: it is assumed that the mean function is 0.
        :param F: (R, N, D, __) - the (samples of the) matrix of GP outputs, where:
                R is the number of Monte Carlo samples,
                N is the numkber of observations and
                D the dimensionality of the covariance matrix.
        :param Y: (N, D) observations
        :return:
            log_det_cov: (R,N) log determinant of the covariance matrix Sigma_n (complexity penalty)
            yt_inv_y: (R,N) (data fit term)
        """
        # Compute Sigma_n (aka AFFA)
        AF = self.A[:, None] * F  # (R, N, D, DoF)
        AFFA = tf.matmul(AF, AF, transpose_b=True)  # (R, N, D, D)

        # additive white noise (Lambda) for numerical precision
        if self.additive_noise:
            n_samples = tf.shape(F)[0]  # could be 1 if making predictions
            dist = tfp.distributions.Gamma(self.q_sigma2inv_conc, self.q_sigma2inv_rate)
            sigma2_inv = dist.sample([n_samples])  # (R, D)
            sigma2_inv = tf.clip_by_value(sigma2_inv, 1e-8, np.inf)

            if self.model_inverse:
                Lambda = sigma2_inv[:, None, :]
            else:
                sigma2 = sigma2_inv**-1.
                Lambda = sigma2[:, None, :]
        else:
            Lambda = 1e-5

        # Compute log determinant of covariance matrix Sigma_n (aka AFFA)
        AFFA = tf.linalg.set_diag(AFFA, tf.linalg.diag_part(AFFA) + Lambda)
        L = tf.linalg.cholesky(AFFA)  # (R, N, D, D)
        log_det_cov = 2 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)), axis=2)  # (R, N)
        if self.model_inverse:
            log_det_cov = - log_det_cov

        # Compute (Y^T affa^-1 Y) term
        if self.model_inverse:
            y_prec = tf.einsum('jk,ijkl->ijl', Y, AFFA)  # (R, N, D)  # j=N, k=D, i=, l=
            yt_inv_y = tf.reduce_sum(y_prec * Y, axis=2)  # (R, N)

        else:
            n_samples = tf.shape(F)[0]  # could be 1 when computing MAP test metric
            Ys = tf.tile(Y[None, :, :, None], [n_samples, 1, 1, 1])  # this is inefficient, but can't get the shapes to play well with cholesky_solve otherwise
            L_solve_y = tf.linalg.triangular_solve(L, Ys, lower=True)  # (R, N, D, 1)
            yt_inv_y = tf.reduce_sum(L_solve_y**2, axis=(2, 3))  # (R, N)

        return log_det_cov, yt_inv_y