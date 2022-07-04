import numpy as np
import gpflow
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from ..models.BaseWishartProcess import WishartProcessBase

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

    def predict_mc(self, X_test, n_samples):
        """
        Returns samples of the covariance matrix $\Sigma_n$ and the mean $\mu$ for each time point

        :param X_test: (N_test,D) input locations to predict covariance matrix over.
        :param n_samples: (int)
        :return:
        """
        A, D, nu, mnu = self.likelihood.A, self.likelihood.D, self.likelihood.nu, self.likelihood.mnu
        N_test, _ = X_test.shape

        # Produce n_samples of F (latent GP points as the input locations X)
        f_sample = self.predict_f_samples(X_test, num_samples=n_samples)
        f_sample = tf.reshape(f_sample, [n_samples, N_test, D, -1])  # (n_samples, N_test, D, nu+mnu)
        
        # Construct Sigma from latent gp's
        AF = A[:, None] * f_sample[:,:,:,:nu]  # (n_samples, N_test, D, nu)
        affa = np.matmul(AF, np.transpose(AF, [0, 1, 3, 2]))  # (n_samples, N_test, D, D)

        #construct Mu from latent gp's
        if mnu == "independent":
            mu = A[:, None] * f_sample[:, :, :, nu:]  # (n_samples, N_test, D, mnu)
            mu = np.sum(mu, axis = -1) # (n_samples, N_test, D)
        elif mnu == "shared":
            mu = A[:, None] * f_sample  # (n_samples, N_test, D, mnu)
            mu = np.sum(mu, axis = -1) # (n_samples, N_test, D)
        elif mnu == "fully_dependent":
            mu = A[:, None] * f_sample[:, :, :, :nu]  # (n_samples, N_test, D, mnu)
            mu = np.sum(mu, axis = -1) # (n_samples, N_test, D)
        elif mnu == "zero":
            mu = np.zeros((N_test, D))
        else:
            print("invalid mnu value, mnu will be set to zero")
            mu = np.zeros(N_test, D)
        if self.likelihood.additive_noise:
            Lambda = self.get_additive_noise(n_samples)
            affa = tf.linalg.set_diag(affa, tf.linalg.diag_part(affa) + Lambda)
        else:
            affa += 1e-6

        return affa , mu

    def predict_map(self, X_test):
        """
        Get mean prediction of Sigma and mu
        :param X_test(N_test, D) input locations to predict covariance matrix over.
        :return: (D,D) mean estimate of covariance
        """
        A, D, nu = self.likelihood.A, self.likelihood.D, self.likelihood.nu
        mnu = self.likelihood.mnu
        N_test, _ = X_test.shape

        # Do not produce n_samples of F (latent GP points as the input locations X)
        mean, var = self.predict_f(X_test)  # (N_test, D*nu)
        mean = tf.reshape(mean, [N_test, D, -1]) # (N_test, D, nu)
        
        AF = A[:, None] * mean[:,:,:nu]
        affa = np.matmul(AF, np.transpose(AF, [0, 2, 1]))  # (N_test, D, D)
        
        #construct Mu from latent gp's
        if mnu == "independent":
            mu = A[:, None] * mean[:, :,nu:]  # (N_test, D, mnu)
            mu = np.sum(mu, axis = -1) # (N_test, D)
        elif mnu == "shared":
            mu = A[:, None] * mean  # (N_test, D, mnu)
            mu = np.sum(mu, axis = -1) # (N_test, D)
        elif mnu == "fully_dependent":
            mu = A[:, None] * mean[:, :,:nu]  # (N_test, D, mnu)
            mu = np.sum(mu, axis = -1) # (N_test, D)
        elif mnu == "zero":
            mu = np.zeros(( N_test, D))
        else:
            print("invalid mnu value, mnu will be set to zero")
            mu = np.zeros(N_test,D)
        if self.likelihood.additive_noise:
            Lambda = self.get_additive_noise(1)
            affa = tf.linalg.set_diag(affa, tf.linalg.diag_part(affa) + Lambda)
        else:
            affa += 1e-6
        return affa, mu