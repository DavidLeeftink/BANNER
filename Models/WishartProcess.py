import numpy as np
import gpflow
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from Likelihoods.WishartProcessLikelihood import WishartLikelihoodBase, FullWishartLikelihood


class WishartProcessBase(gpflow.models.SVGP):
    # Upgrade guide SVGP Model:
    # https://gpflow.readthedocs.io/en/master/notebooks/gpflow2_upgrade_guide.html#SVGP-Initialiser

    def __init__(self, kernel, likelihood=None, D=1, DoF=None, inducing_variable=None):
        DoF = D if DoF is None else DoF
        likelihood = FullWishartLikelihood(D, DoF, R=10) if likelihood is None else likelihood
        self.cov_dim = likelihood.D # for now, use the input dimensionality as covariance size.

        super().__init__(kernel=kernel,
                         likelihood=likelihood,
                         num_latent_gps=int(D*DoF),
                         inducing_variable=inducing_variable)

    def construct_predictive_density(self):
        """
        To do: confirm behaviour of this function is comparable to implementation. (which seemed to contain a lot of duplicate code from the likelihood class)
        :param
        """
        # create placeholders with yet unspecified value for N
        D, DoF = self.likelihood.D, self.likelihood.DoF
        self.X_new = tf.compat.v1.placeholder(dtype=tf.float64, shape=[None, 1])
        self.Y_new = tf.compat.v1.placeholder(dtype=tf.float64, shape=[None, D])
        self.R = tf.compat.v1.placeholder(dtype=tf.int64, shape=[])

        #obtain predictive function values
        F_mean, F_var = self.predict_f(self.X_new) # build_predict -> predict_f
        # # todo: check if above is correct # (N_new, num_latent); e.g., num_latent = D * nu
        N_new = tf.shape(F_mean)[0]
        self.logp = self.likelihood.variational_expectations(F_mean, F_var)

    def mcmc_predict_density(self, X_test, Y_test, n_samples):
        """:param
        """
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError


class FullCovarianceWishartProcess(WishartProcessBase):
    def __init__(self, kernel, **kwargs):
        super().__init__(kernel)

    def build_prior_KL(self):
        """
        Function that adds diagonal likelihood noise to the default stochastic variational KL prior.
        """
        KL = super().build_prior_KL()
        p_dist = tfd.Gamma(self.likelihood.p_sigma2inv_conc, rate=self.likelihood.p_sigma2inv_rate)
        q_dist = tfd.Gamma(self.likelihood.q_sigma2inv_rate, rate=self.likelihood.q_sigma2inv_rate)
        self.KL_gamma = tf.reduce_sum(q_dist.kl_divergence(p_dist))
        KL += self.KL_gamma
        return KL

    # def mcmc_predict_density(self, X_test, Y_test, n_samples):
    #     params = self.predict(X_test)
    #     mu, s2 = params['mu'], params['s2']
    #     scale_diag = params['scale_diag']
    #
    #     N_new, D, DoF = mu.shape
    #     F_samps = np.random.randn(n_samples, N_new, D, DoF) * np.sqrt(s2) + mu  # (n_samples, N_new, D, nu)
    #     AF = scale_diag[:, None] * F_samps  # (n_samples, N_new, D, nu)
    #     affa = np.matmul(AF, np.transpose(AF, [0, 1, 3, 2]))  # (n_samples, N_new, D, D)
    #
    #     if self.likelihood.approx_wishart:
    #         sigma2inv_conc = params['sigma2inv_conc']
    #         sigma2inv_rate = params['sigma2inv_rate']
    #         sigma2inv_samps = np.random.gamma(sigma2inv_conc, scale=1.0 / sigma2inv_rate,
    #                                           size=[n_samples, D])  # (n_samples, D)
    #
    #         if self.likelihood.model_inverse:
    #             lam = np.apply_along_axis(np.diag, axis=0, arr=sigma2inv_samps)  # (n_samples, D, D)
    #         else:
    #             lam = np.apply_along_axis(np.diag, axis=0, arr=sigma2inv_samps ** -1.0)
    #         affa = affa + lam[:, None, :, :]  # (n_samples, N_new, D, D)
    #     return affa







