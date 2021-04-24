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
    def __init__(self, kernel, likelihood=None, D=1, DoF=None, inducing_variable=None, q_mu=None, q_sqrt=None):
        """
        :param kernel (gpflow.Kernel object)
        :param likelihood (gpflow.likelihood object)
        :param D (int) Covariance matrix dimension
        :param DoF (int) Degrees of freedom
        :param inducing_variable ()
        """
        DoF = D if DoF is None else DoF
        likelihood = WishartLikelihood(D, DoF, R=10) if likelihood is None else likelihood

        super().__init__(kernel=kernel,
                         likelihood=likelihood,
                         num_latent_gps=int(D*DoF),
                         inducing_variable=inducing_variable,
                         q_mu=q_mu,
                         q_sqrt=q_sqrt)

    def construct_predictive_density(self):
        """
        to do: confirm if this can be removed in TF 2.
        """
        # create placeholders with yet unspecified value for N
        D, DoF = self.likelihood.D, self.likelihood.DoF
        self.X_new = tf.compat.v1.placeholder(dtype=tf.float64, shape=[None, 1])
        self.Y_new = tf.compat.v1.placeholder(dtype=tf.float64, shape=[None, D])
        self.R = tf.compat.v1.placeholder(dtype=tf.int64, shape=[])

        # obtain predictive function values
        F_mean, F_var = self.predict_f(self.X_new) # build_predict -> predict_f # (N_new, num_latent)
        # todo: check if above is correct
        N_new = tf.shape(F_mean)[0]
        self.logp = self.likelihood.variational_expectations(F_mean, F_var)

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
        A, D, DoF = self.likelihood.A, self.likelihood.D, self.likelihood.DoF
        N_test, _ = X_test.shape

        # Produce n_samples of F (latent GP points as the input locations X)
        mu, var = self.predict_f(X_test) # (N_test, D*DoF)
        W = tf.dtypes.cast(tf.random.normal([n_samples, N_test, int(D * DoF)]), tf.float64)
        f_sample = W * var**0.5 + mu
        f_sample = tf.reshape(f_sample, [n_samples, N_test, D, -1]) # (n_samples, N_test, D, DoF)

        # Construct Sigma from latent gp's
        AF = A[:, None] * f_sample  # (n_samples, N_test, D, DoF)
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
        A, D, DoF = self.likelihood.A, self.likelihood.D, self.likelihood.DoF
        N_test, _ = X_test.shape

        # Produce n_samples of F (latent GP points as the input locations X)
        mu, var = self.predict_f(X_test)  # (N_test, D*DoF)
        mu = tf.reshape(mu, [N_test, D, -1]) # (N_test, D, DoF)
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

    def predict_map(self, X_test):
        """
        Get mean prediction
        :param X_test(N_test, D) input locations to predict covariance matrix over.
        :return: params (dictionary) contains the likelihood parameters, monitored by tensorboard.
        """
        A, D, DoF = self.likelihood.A, self.likelihood.D, self.likelihood.DoF
        N_test, _ = X_test.shape

        # Produce n_samples of F (latent GP points as the input locations X)
        mu, var = self.predict_f(X_test)  # (N_test, D*DoF)
        mu = tf.reshape(mu, [N_test, D, -1]) # (N_test, D, DoF)
        AF = A[:, None] * mu
        affa = np.matmul(AF, np.transpose(AF, [0, 2, 1]))  # (N_test, D, D)
        if self.likelihood.additive_noise:
            Lambda = self.get_additive_noise(1)
            affa = tf.linalg.set_diag(affa, tf.linalg.diag_part(affa) + Lambda)
        else:
            affa += 1e-6
        return affa
