import numpy as np
import gpflow
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from gpflow.likelihoods.base import ScalarLikelihood
from gpflow.utilities import positive
from gpflow import Parameter


class WishartLikelihoodBase(ScalarLikelihood):
    """
    Abstract class for all Wishart Processes likelihoods.
    Class written by Creighton Heaukulani and Mark van der Wilk, and is adapted for gpflow 2.
    """
    def __init__(self, D, nu, R=10, model_inverse=True, additive_noise=True, multiple_observations=False, **kwargs):
        """
        :param D (int) Covariance matrix dimension
        :param nu (int) Degrees of freedom
        :param R (int) Number of monte carlo samples used to approximate reparameterized gradients.
        :param inverse (bool) Use inverse Wishart Process if true, otherwise standard Wishart Process.
        :param additive_noise (bool) Use additive white noise model likelihood if true.
        :param multiple_observations (bool) At each timepoint, multiple observations are available. (i.e. the data is TxNxD)
        """
        super().__init__()  # todo: check likelihoods' specification of dimensions
        self.D = D
        self.nu = nu
        self.R = R
        self.model_inverse = model_inverse
        self.additive_noise = additive_noise
        self.multiple_observations = multiple_observations

    def variational_expectations(self, f_mean, f_cov, Y):
        """
        Calculate log p(Y | variational parameters)

        :param f_mean: (N, D*nu), mean parameters of latent GP points F
        :param f_cov: (N, D*nu), covariance parameters of latent GP points F
        :param Y: (N, D) or (T,N,D), observations
        :return logp: (N,), log probability density of the data.
        where N is the minibatch size, D the covariance matrix dimension and nu the degrees of freedom
        """
        _, latent_dim = f_mean.shape
        print(f_mean.shape)
        cov_dim = self.cov_dim
        N = Y.shape[0] if not self.multiple_observations else Y.shape[1]

        # Produce R samples of F (latent GP points at the input locations X).
        # TF automatically differentiates through this.
        W = tf.dtypes.cast(tf.random.normal(shape=[self.R, N, latent_dim]), tf.float64)
        f_sample = W * f_cov**0.5 + f_mean

        f_sample = tf.reshape(f_sample, [self.R, N, self.cov_dim, -1])

        # compute the mean of the likelihood
        logp = self._log_prob(f_sample, Y) #(N,)
        return logp

    def _log_prob(self, F, Y): # (R,N) -> (N)
        if self.multiple_observations:
            logps = []
            for t in range(Y.shape[0]):
                Y_t = Y[t]
                logps.append(tf.math.reduce_mean(self._scalar_log_prob(F,Y_t), axis=0)) # (R,N) -> (N,)
            return tf.math.reduce_sum(logps, axis=0) # (T,N) -> (N,)
        else:
            return tf.math.reduce_mean(self._scalar_log_prob(F, Y), axis=0) # take mean across D dimension

    def _scalar_log_prob(self, F, Y):
        """
        Log probability of covariance matrix Sigma_n = A F_n F_n^time_window A^time_window
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


class WishartLikelihood(WishartLikelihoodBase):
    """
    Concrete class for the full covariance likelihood models.
    The code is written by Heaukulani-van der Wilk (see references above)
    """
    def __init__(self, D, nu, A=None, **kwargs):
        """
        :param D (int) Dimensionality of covariance matrix
        :param nu (int) degrees of freedom
        :param A (DxD matrix) scale matrix. Default is a DxD identity matrix.
        """
        assert nu >= D, "Degrees of freedom must be larger or equal than the dimensionality of the covariance matrix"
        super().__init__(D, nu, **kwargs)
        self.cov_dim = D

        # this case assumes a square scale matrix, and it must lead with dimension D
        self.A = A if A is not None else Parameter(np.ones(self.D), transform=positive(), dtype=tf.float64)
        gpflow.set_trainable(self.A, False)

        if self.additive_noise:
            self.p_sigma2inv_conc = Parameter(0.1, transform=positive(), dtype=tf.float64)
            self.p_sigma2inv_rate = Parameter(0.0001, transform=positive(), dtype=tf.float64)
            self.q_sigma2inv_conc = Parameter(0.1 * np.ones(self.D), transform=positive(), dtype=tf.float64)
            self.q_sigma2inv_rate = Parameter(0.0001 * np.ones(self.D), transform=positive(), dtype=tf.float64)

    def make_gaussian_components(self, F, Y):
        """
        An auxiliary function for logp that returns the complexity pentalty and the data fit term.
        Note: it is assumed that the mean function is 0.
        :param F: (R, N, D, __) - the (samples of the) matrix of GP outputs, where:
                R is the number of Monte Carlo samples,
                N is the numkber of observations and
                D the dimensionality of the covariance matrix.
        :param Y: (N, D) Tensor. observations
        :return:
            log_det_cov: (R,N) log determinant of the covariance matrix Sigma_n (complexity penalty)
            yt_inv_y: (R,N) (data fit term)
        """
        # Compute Sigma_n (aka AFFA)
        AF = self.A[:, None] * F  # (R, N, D, nu)
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

        # Compute (Y^time_window affa^-1 Y) term
        if self.model_inverse:
            y_prec = tf.einsum('jk,ijkl->ijl', Y, AFFA)  # (R, N, D)  # j=N, k=D, i=, l=
            yt_inv_y = tf.reduce_sum(y_prec * Y, axis=2)  # (R, N)

        else:
            n_samples = tf.shape(F)[0]  # could be 1 when computing MAP test metric
            Ys = tf.tile(Y[None, :, :, None], [n_samples, 1, 1, 1])  # this is inefficient, but can't get the shapes to play well with cholesky_solve otherwise
            L_solve_y = tf.linalg.triangular_solve(L, Ys, lower=True)  # (R, N, D, 1)
            yt_inv_y = tf.reduce_sum(L_solve_y**2, axis=(2, 3))  # (R, N)

        return log_det_cov, yt_inv_y


class FactorizedWishartLikelihood(WishartLikelihoodBase):

    def __init__(self, D, nu, n_factors, A=None, **kwargs):
        """
        :param D (int) Covariance matrix dimension
        :param nu (int) Degrees of freedom
        :param n_factors (int) Dimensionality of factorized covariance matrix.
        :param R (int) Number of monte carlo samples used to approximate reparameterized gradients.
        :param inverse (bool) Use inverse Wishart Process if true, otherwise standard Wishart Process.
        """
        super().__init__(D, nu, additive_noise=True, **kwargs)  # todo: check likelihoods' specification of dimensions
        self.D = D
        self.cov_dim = n_factors
        self.nu = nu
        self.n_factors = n_factors

        # no such thing as a non-full scale matrix in this case
        self.A = A if A is not None else Parameter(np.ones((D, n_factors)), transform=positive(), dtype=tf.float64)
        gpflow.set_trainable(self.A, True)

        # all factored models are approximate models
        self.p_sigma2inv_conc = Parameter(0.1, transform=positive(), dtype=tf.float64)
        self.p_sigma2inv_rate = Parameter(0.0001, transform=positive(), dtype=tf.float64)
        self.q_sigma2inv_conc = Parameter(0.1 * np.ones(self.D), transform=positive(), dtype=tf.float64)
        self.q_sigma2inv_rate = Parameter(0.0001 * np.ones(self.D), transform=positive(), dtype=tf.float64)

    def make_gaussian_components(self, F, Y):
        """
        In the case of the factored covariance matrices, we should never directly represent the covariance or precision
        matrix. The following computation makes use of the matrix inversion formula(s).
        Function written entirely by Creighton Heaukulani and Mark van der Wilk.

        :param F: (R, N, K, nu2) - the (samples of the) matrix of GP outputs.
        :param Y: (N, D)
        :return:
        """
        print('pre AF', self.A.shape, F.shape, Y.shape)
        # k = D = 3
        # l = n_factors = 2
        # ,
        # i = R = 10
        # j = N = 100
        # m = nu = 4
        # (2,3) (10,100,2,4) -> (10,100,3,6)
        #

        AF = tf.einsum('kl,ijlm->ijkm', self.A, F)  # (S, N, D, nu*2) # todo: why did the doc say 2nu here? the final shape is still nu as only l (n_factors) is summed and multiplied out
        print('AF shape', AF.shape)
        n_samples = tf.shape(F)[0]  # could be 1 if making predictions
        dist = tfd.Gamma(self.q_sigma2inv_conc, self.q_sigma2inv_rate)
        sigma2_inv = dist.sample([n_samples])  # (S, D)
        sigma2_inv = tf.clip_by_value(sigma2_inv, 1e-8, np.inf)
        sigma2 = sigma2_inv ** -1.0

        # if tf.is_tensor(Y):
        #     Y.set_shape([None, self.D])

        y_Sinv_y = tf.reduce_sum((Y ** 2.0) * sigma2_inv[:, None, :], axis=2)  # (S, N)

        if self.model_inverse:
            # no inverse necessary for Gaussian exponent
            SAF = sigma2[:, None, :, None] * AF  # (S, N, D, nu2)
            faSaf = tf.matmul(AF, SAF, transpose_a=True)  # (S, N, nu2, nu2)
            faSaf = tf.linalg.set_diag(faSaf, tf.linalg.diag_part(faSaf) + 1.0)
            L = tf.linalg.cholesky(faSaf)  # (S, N, nu2, nu2)
            log_det_cov = tf.reduce_sum(tf.math.log(sigma2), axis=1)[:, None] \
                           - 2 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)), axis=2)  # (S, N)
                # note: first line had negative because we needed log(s2^-1) and then another negative for |precision|

            yaf_or_afy = tf.einsum('jk,ijkl->ijl', Y, AF)  # (S, N, nu2)
            yt_inv_y = y_Sinv_y + tf.reduce_sum(yaf_or_afy ** 2, axis=2)  # (S, N)

        else:
            # Wishart case: take the inverse to create Gaussian exponent
            SinvAF = sigma2_inv[:, None, :, None] * AF  # (S, N, D, nu^2)
            faSinvaf = tf.matmul(AF, SinvAF, transpose_a=True)  # (S, N, nu2, nu2), computed efficiently, O(S * N * n_factors^2 * D)

            faSinvaf = tf.linalg.set_diag(faSinvaf, tf.linalg.diag_part(faSinvaf) + 1.0)
            L = tf.linalg.cholesky(faSinvaf)  # (S, N, nu2, nu2)
            log_det_cov = tf.reduce_sum(tf.math.log(sigma2), axis=1)[:, None] \
                           + 2 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)), axis=2)  # (S, N), just log |AFFA + S| (no sign)

            ySinvaf_or_afSinvy = tf.einsum('jk,ijkl->ijl', Y, SinvAF)  # (S, N, nu2)
            L_solve_ySinvaf = tf.linalg.triangular_solve(L, ySinvaf_or_afSinvy[:, :, :, None], lower=True)  # (S, N, nu2, 1)
            ySinvaf_inv_faSinvy = tf.reduce_sum(L_solve_ySinvaf ** 2.0, axis=(2, 3))  # (S, N)
            yt_inv_y = y_Sinv_y - ySinvaf_inv_faSinvy  # (S, N), this is Y^time_window (AFFA + S)^-1 Y

        return log_det_cov, yt_inv_y

    # def make_gaussian_components(self, F, Y):
    #     """
    #     In the case of the factored covariance matrices, we should never directly represent the covariance or precision
    #     matrix. The following computation makes use of the matrix inversion formula(s).
    #     :param F: (S, N, n_factors, nu2) - the (samples of the) matrix of GP outputs.
    #     :param Y: (N, D)
    #     :return:
    #     """
    #     AF = tf.einsum('kl,ijlm->ijkm', self.scale, F)  # (S, N, D, nu2)
    #
    #     n_samples = tf.shape(F)[0]  # could be 1 if making predictions
    #     dist = tf.distributions.Gamma(self.q_sigma2inv_conc, self.q_sigma2inv_rate)
    #     sigma2_inv = dist.sample([n_samples])  # (S, D)
    #     sigma2_inv = tf.clip_by_value(sigma2_inv, 1e-8, np.inf)
    #     sigma2 = sigma2_inv ** -1.0
    #
    #     Y.set_shape([None, self.D])  # in GPflow 1.0 I didn't need to do this
    #     y_Sinv_y = tf.reduce_sum((Y ** 2.0) * sigma2_inv[:, None, :], axis=2)  # (S, N)
    #
    #     if self.model_inverse:
    #         # no inverse necessary for Gaussian exponent
    #         SAF = sigma2[:, None, :, None] * AF  # (S, N, D, nu2)
    #         faSaf = tf.matmul(AF, SAF, transpose_a=True)  # (S, N, nu2, nu2)
    #         faSaf = tf.matrix_set_diag(faSaf, tf.matrix_diag_part(faSaf) + 1.0)
    #         L = tf.cholesky(faSaf)  # (S, N, nu2, nu2)
    #         log_det_cov = tf.reduce_sum(tf.log(sigma2), axis=1)[:, None] \
    #                       - 2 * tf.reduce_sum(tf.log(tf.matrix_diag_part(L)), axis=2)  # (S, N)
    #         # note: first line had negative because we needed log(s2^-1) and then another negative for |precision|
    #
    #         yaf_or_afy = tf.einsum('jk,ijkl->ijl', Y, AF)  # (S, N, nu2)
    #         yt_inv_y = y_Sinv_y + tf.reduce_sum(yaf_or_afy ** 2, axis=2)  # (S, N)
    #
    #     else:
    #         # Wishart case: take the inverse to create Gaussian exponent
    #         SinvAF = sigma2_inv[:, None, :, None] * AF  # (S, N, D, nu2)
    #         faSinvaf = tf.matmul(AF, SinvAF,
    #                              transpose_a=True)  # (S, N, nu2, nu2), computed efficiently, O(S * N * n_factors^2 * D)
    #
    #         faSinvaf = tf.matrix_set_diag(faSinvaf, tf.matrix_diag_part(faSinvaf) + 1.0)
    #         L = tf.cholesky(faSinvaf)  # (S, N, nu2, nu2)
    #         log_det_cov = tf.reduce_sum(tf.log(sigma2), axis=1)[:, None] \
    #                       + 2 * tf.reduce_sum(tf.log(tf.matrix_diag_part(L)),
    #                                           axis=2)  # (S, N), just log |AFFA + S| (no sign)
    #
    #         ySinvaf_or_afSinvy = tf.einsum('jk,ijkl->ijl', Y, SinvAF)  # (S, N, nu2)
    #         L_solve_ySinvaf = tf.matrix_triangular_solve(L, ySinvaf_or_afSinvy[:, :, :, None],
    #                                                      lower=True)  # (S, N, nu2, 1)
    #         ySinvaf_inv_faSinvy = tf.reduce_sum(L_solve_ySinvaf ** 2.0, axis=(2, 3))  # (S, N)
    #         yt_inv_y = y_Sinv_y - ySinvaf_inv_faSinvy  # (S, N), this is Y^time_window (AFFA + S)^-1 Y
    #
    #     return log_det_cov, yt_inv_y