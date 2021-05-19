import numpy as np
import gpflow
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from Likelihoods.WishartProcessLikelihood import WishartLikelihoodBase, WishartLikelihood

class WishartProcessBase(gpflow.models.SVGP):
    """
    Wrapper around gpflow's SVGP class, with added functionaility for estimating the covariance matrix.
    Class written by Creighton Heaukulani and Mark van der Wilk, and is adapted for gpflow 2.
    """
    def __init__(self, kernel, likelihood=None, D=1, nu=None, inducing_variable=None, q_mu=None, q_sqrt=None):
        """
        :param kernel (gpflow.Kernel object)
        :param likelihood (gpflow.likelihood object)
        :param D (int) Covariance matrix dimension
        :param nu (int) Degrees of freedom
        :param inducing_variable ()
        """
        nu = D if nu is None else nu
        likelihood = WishartLikelihood(D, nu, R=10) if likelihood is None else likelihood

        super().__init__(kernel=kernel,
                         likelihood=likelihood,
                         num_latent_gps=int(D * nu),
                         inducing_variable=inducing_variable,
                         q_mu=q_mu,
                         q_sqrt=q_sqrt)


    def predict_mc(self, X_test, Y_test, n_samples):
        """
        Returns monte carlo samples of the covariance matrix $\Sigma$
        Abstract method, should be implemented by concrete class.

        :param X_test (N_test, D) input locations to predict covariance matrix over.
        :param Y_test (N_test, D) observations to predict covariance matrix over.
        :param n_samples (int) number of samples to estimate covariance matrix at each time point.
        :return Sigma (n_samples, N_test, D, D) covariance matrix sigma
        """
        raise NotImplementedError

    def predict_map(self, X_test):
        """
        Returns MAP estimate of the covariance matrix $\Sigma$
        Abstract method, should be implemented by concrete class.

        :param X_test (N_test, D) input locations to predict covariance matrix over.
        :param Y_test (N_test, D) observations to predict covariance matrix over.
        :return Sigma (N_test, D, D) covariance matrix sigma
        """
        raise NotImplementedError

    def get_additive_noise(self, n_samples):
        """
        Get n samples of white noise to add on diagonal matrix
        :param n_samples
        :return Lambda Additive white noise
        """
        sigma2inv_conc = self.likelihood.q_sigma2inv_conc
        sigma2inv_rate = self.likelihood.q_sigma2inv_rate

        dist = tfd.Gamma(sigma2inv_conc, sigma2inv_rate)
        sigma2_inv = dist.sample([n_samples])  # (R, D)
        sigma2_inv = tf.clip_by_value(sigma2_inv, 1e-8, np.inf)

        if self.likelihood.model_inverse:
            Lambda = sigma2_inv[:, None, :]
        else:
            sigma2 = np.power(sigma2_inv, -1.0)
            Lambda = sigma2[:, None, :]
        if Lambda.shape[0]==1:
            Lambda = np.reshape(Lambda, -1)
        return Lambda


class WishartProcess(WishartProcessBase):
    """
    Concrete model that implements the (inverse) Wishart Process with the full covariance matrix.
    """
    def __init__(self, kernel, likelihood, **kwargs):
        super().__init__(kernel, likelihood, **kwargs)

    def build_prior_KL(self):
        """
        Function that adds diagonal likelihood noise to the default stochastic variation¡al KL prior.

        :return KL () Kullback-Leibler divergence including diagonal white noise.
        """
        KL = super().build_prior_KL()
        if self.likelihood.additive_noise:
            p_dist = tfd.Gamma(self.likelihood.p_sigma2inv_conc, rate=self.likelihood.p_sigma2inv_rate)
            q_dist = tfd.Gamma(self.likelihood.q_sigma2inv_rate, rate=self.likelihood.q_sigma2inv_rate)
            self.KL_gamma = tf.reduce_sum(q_dist.kl_divergence(p_dist))
            KL += self.KL_gamma
        return KL

    def predict_mc(self, X_test, Y_test, n_samples):
        """
        Returns samples of the covariance matrix $\Sigma_n$ for each time point

        :param X_test: (N_test,D) input locations to predict covariance matrix over.
        :param Y_test: (N_test,D) observations to predict covariance matrix over.
        :param n_samples: (int)
        :return:
        """
        A, D, nu = self.likelihood.A, self.likelihood.D, self.likelihood.nu
        N_test, _ = X_test.shape

        # Produce n_samples of F (latent GP points as the input locations X)
        mu, var = self.predict_f(X_test) # (N_test, D*nu)
        W = tf.dtypes.cast(tf.random.normal([n_samples, N_test, int(D * nu)]), tf.float64)
        f_sample = W * var**0.5 + mu
        f_sample = tf.reshape(f_sample, [n_samples, N_test, D, -1]) # (n_samples, N_test, D, nu)

        # Construct Sigma from latent gp's
        AF = A[:, None] * f_sample  # (n_samples, N_test, D, nu)
        affa = np.matmul(AF, np.transpose(AF, [0, 1, 3, 2]))  # (n_samples, N_test, D, D)

        if self.likelihood.additive_noise:
            Lambda = self.get_additive_noise(n_samples)
            affa = tf.linalg.set_diag(affa, tf.linalg.diag_part(affa) + Lambda)
        else:
            affa += 1e-6

        return affa

    def predict_map(self, X_test):
        """
        Get mean prediction
        :param X_test(N_test, D) input locations to predict covariance matrix over.
        :return: (D,D) mean estimate of covariance
        """
        A, D, nu = self.likelihood.A, self.likelihood.D, self.likelihood.nu
        N_test, _ = X_test.shape

        # Produce n_samples of F (latent GP points as the input locations X)
        mu, var = self.predict_f(X_test)  # (N_test, D*nu)
        mu = tf.reshape(mu, [N_test, D, -1]) # (N_test, D, nu)
        AF = A[:, None] * mu
        affa = np.matmul(AF, np.transpose(AF, [0, 2, 1]))  # (N_test, D, D)
        if self.likelihood.additive_noise:
            Lambda = self.get_additive_noise(1)
            affa = tf.linalg.set_diag(affa, tf.linalg.diag_part(affa) + Lambda)
        else:
            affa += 1e-6
        return affa


class FactorizedWishartModel(WishartProcessBase):
    def __init__(self, kernel, likelihood, **kwargs):
        super().__init__(kernel, likelihood, **kwargs)

    def build_prior_KL(self):
        """
        Function that adds diagonal likelihood noise to the default stochastic variation¡al KL prior.

        :return KL () Kullback-Leibler divergence including diagonal white noise.
        """
        KL = super().build_prior_KL()
        p_dist = tfd.Gamma(self.likelihood.p_sigma2inv_conc, rate=self.likelihood.p_sigma2inv_rate)
        q_dist = tfd.Gamma(self.likelihood.q_sigma2inv_rate, rate=self.likelihood.q_sigma2inv_rate)
        self.KL_gamma = tf.reduce_sum(q_dist.kl_divergence(p_dist))
        KL += self.KL_gamma
        return KL

    def predict_mc(self, X_test, Y_test, n_samples):
        """
        Returns samples of the covariance matrix $\Sigma_n$ for each time point

        :param X_test: (N_test,D) input locations to predict covariance matrix over.
        :param Y_test: (N_test,D) observations to predict covariance matrix over.
        :param n_samples: (int)
        :return: (n_samples, K, K)
         """
        A, D, nu = self.likelihood.A, self.likelihood.D, self.likelihood.nu
        N_test, _ = X_test.shape

        # Produce n_samples of F (latent GP points as the input locations X)
        mu, var = self.predict_f(X_test)  # (N_test, D*nu)
        W = tf.dtypes.cast(tf.random.normal([n_samples, N_test, int(D * nu)]), tf.float64)
        f_sample = W * var ** 0.5 + mu
        f_sample = tf.reshape(f_sample, [n_samples, N_test, D, -1])  # (n_samples, N_test, D, nu)

        # Construct Sigma from latent gp's
        AF = A[:, None] * f_sample  # (n_samples, N_test, D, nu)
        affa = np.matmul(AF, np.transpose(AF, [0, 1, 3, 2]))  # (n_samples, N_test, D, D)

        if self.likelihood.additive_noise:
            Lambda = self.get_additive_noise(n_samples)
            affa = tf.linalg.set_diag(affa, tf.linalg.diag_part(affa) + Lambda)
        else:
            affa += 1e-6

        return affa

    def predict_map(self, X_test):
        """
        Get mean prediction
        :param X_test(N_test, D) input locations to predict covariance matrix over.
        :return: params (dictionary) contains the likelihood parameters, monitored by tensorboard.
        """
        A, D, nu = self.likelihood.A, self.likelihood.D, self.likelihood.nu
        A, D, nu = self.likelihood.A, self.likelihood.D, self.likelihood.nu
        N_test, _ = X_test.shape

        # Produce n_samples of F (latent GP points as the input locations X)
        mu, var = self.predict_f(X_test)  # (N_test, D*nu)
        mu = tf.reshape(mu, [N_test, D, -1])  # (N_test, D, nu)
        AF = A[:, None] * mu
        affa = np.matmul(AF, np.transpose(AF, [0, 2, 1]))  # (N_test, D, D)
        if self.likelihood.additive_noise:
            Lambda = self.get_additive_noise(1)
            affa = tf.linalg.set_diag(affa, tf.linalg.diag_part(affa) + Lambda)
        else:
            affa += 1e-6
        return affa

# Heakulaini:
# class FactoredCovLikelihood(DynamicCovarianceBaseLikelihood):
#     """
#     Concrete class for factored covariance models.
#     """
#     def __init__(self, D, n_mc_samples, n_factors, heavy_tail, model_inverse, nu=None, dof=2.5):
#         """
#         :param D: The dimensionality of the covariance matrix being constructed with the multi-output GPs.
#         :param n_mc_samples: The number of Monte Carlo samples to use to approximate the reparameterized gradients.
#         :param n_factors: int - The dimensionality of the constructed Wishart matrix, i.e., the leading size of the
#             array of GPs.
#         :param heavy_tail: bool - If True, use the multivariate-t distribution emission model.
#         :param model_inverse: bool - If True, we are modeling the inverse of the Covariance matrix with a Wishart
#             distribution, i.e., this corresponds to an inverse Wishart process model.
#         :param nu: int - The degrees of freedom of the Wishart distributed matrix being constructed by the multi-output
#             GPs. Since that matrix has dimension equal to 'n_factors', we must have nu >= n_factors to ensure the
#             Wishart matrix remains nonsingular.
#         :param dof: float - If 'heavy_tail' is True, then this is used to initialize the multivariate-t distribution
#             degrees of freedom parameter, otherwise, it is ignored.
#         """
#
#         nu = n_factors if nu is None else nu
#         if nu < n_factors:
#             raise Exception("Wishart DOF must be >= n_factors.")
#
#         super().__init__(D, cov_dim=n_factors, nu=nu, n_mc_samples=n_mc_samples, model_inverse=model_inverse,
#                          heavy_tail=heavy_tail, dof=dof)
#
#         self.n_factors = n_factors
#         self.model_inverse = model_inverse
#
#         # no such thing as a non-full scale matrix in this case
#         A = np.ones([self.D, self.n_factors])
#         self.scale = Parameter(A, transform=transforms.positive, dtype=settings.float_type)
#
#         # all factored models are approximate models
#         self.p_sigma2inv_conc = Parameter(0.1, transform=transforms.positive, dtype=settings.float_type)
#         self.p_sigma2inv_rate = Parameter(0.0001, transform=transforms.positive, dtype=settings.float_type)
#         self.q_sigma2inv_conc = Parameter(0.1 * np.ones(self.D), transform=transforms.positive, dtype=settings.float_type)
#         self.q_sigma2inv_rate = Parameter(0.0001 * np.ones(self.D), transform=transforms.positive, dtype=settings.float_type)
#
#