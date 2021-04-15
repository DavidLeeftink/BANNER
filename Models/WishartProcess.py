import numpy as np
import gpflow
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from Likelihoods.WishartProcessLikelihood import WishartLikelihoodBase, FullWishartLikelihood

class WishartProcessBase(gpflow.models.SVGP):
    """
    Wrapper around gpflow's SVGP class, with added functionaility for estimating the covariance matrix.
    Class written by Creighton Heaukulani and Mark van der Wilk, and is adapted for gpflow 2.
    """
    def __init__(self, kernel, likelihood=None, D=1, DoF=None, inducing_variable=None):
        """
        :param kernel (gpflow.Kernel object)
        :param likelihood (gpflow.likelihood object)
        :param D (int) Covariance matrix dimension
        :param DoF (int) Degrees of freedom
        :param inducing_variable ()
        """
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

    def mcmc_predict_density(self, X_test, Y_test, n_samples):
        """
        Returns samples of the covariance matrix $\Sigma_n$ for each time point
        Abstract method, should be implemented by concrete class.
        todo: rename this to mc_predict_Sigma (?) because no markov chain is involved, it is just monte carlo

        :param X_test (N_test, D) input locations to predict covariance matrix over.
        :param Y_test (N_test, D) observations to predict covariance matrix over.
        :param n_samples (int) number of samples to estimate covariance matrix at each time point.
        :return Sigma (n_samples, N_test, D, D) covariance matrix sigma
        """
        raise NotImplementedError

    def predict(self, X_test):
        """
        Not yet clear what this does. It appears to be a helper function for the
        Abstract method, should be implemented by concrete class.
        :param X_test(N_test, D) input locations to predict covariance matrix over.
        """
        raise NotImplementedError


class FullCovarianceWishartProcess(WishartProcessBase):
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
        p_dist = tfd.Gamma(self.likelihood.p_sigma2inv_conc, rate=self.likelihood.p_sigma2inv_rate)
        q_dist = tfd.Gamma(self.likelihood.q_sigma2inv_rate, rate=self.likelihood.q_sigma2inv_rate)
        self.KL_gamma = tf.reduce_sum(q_dist.kl_divergence(p_dist))
        KL += self.KL_gamma
        return KL

    def mcmc_predict_density(self, X_test, Y_test, n_samples):
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

        AF = A[:, None] * f_sample  # (n_samples, N_test, D, DoF)
        affa = np.matmul(AF, np.transpose(AF, [0, 1, 3, 2]))  # (n_samples, N_test, D, D)

        if self.likelihood.additive_noise:
            sigma2inv_conc = self.likelihood.q_sigma2inv_conc
            sigma2inv_rate = self.likelihood.q_sigma2inv_rate
            # sigma2inv_samps = np.random.gamma(sigma2inv_conc, scale=1.0 / sigma2inv_rate, #todo: why is this 1/ invrate, but in the likelihood only the invrate?
            #                                   size=[n_samples, D])  # (n_samples, D)
            #
            # if self.likelihood.model_inverse:
            #     diagonal_noise = np.apply_along_axis(np.diag, axis=0, arr=sigma2inv_samps)  # (n_samples, D, D)
            # else:
            #     diagonal_noise = np.apply_along_axis(np.diag, axis=0, arr=sigma2inv_samps ** -1.0)
            #affa = affa + diagonal_noise[:, None, :, :]  # (n_samples, N_new, D, D)
            dist = tfd.Gamma(sigma2inv_conc, sigma2inv_rate)
            sigma2_inv = dist.sample([n_samples])  # (R, D)
            sigma2_inv = tf.clip_by_value(sigma2_inv, 1e-8, np.inf)

            if self.likelihood.model_inverse:
                Lambda = sigma2_inv[:, None, :]
            else:
                sigma2 = np.power(sigma2_inv, -1.0)
                Lambda = sigma2[:, None, :]
            affa = tf.linalg.set_diag(affa, tf.linalg.diag_part(affa) + Lambda)

        else:
            affa += 1e-6
        return affa

    def predict_MAP(self, X_test):
        """

        :param X_test(N_test, D) input locations to predict covariance matrix over.
        :return: params (dictionary) contains the likelihood parameters, monitored by tensorboard.
        """
        #todo: confirm this is dead code
        assert 1==2, "function currently not used. Will likely be removed"
        sess = self.enquire_session()
        mu, s2 = sess.run([self.F_mean_new, self.F_var_new], # todo: self.F_mean_new and self.F_var_new are not yet implemented
                          feed_dict={self.X_new: X_test})  # (N_new, D, DoF), (N_new, D, DoF)
        A = self.likelihood.A.read_value(sess)  # (D,)
        params = dict(mu=mu, s2=s2, scale_diag=A)

        if self.likelihood.additive_noise:
            sigma2inv_conc = self.likelihood.q_sigma2inv_conc.read_value(sess)  # (D,)
            sigma2inv_rate = self.likelihood.q_sigma2inv_rate.read_value(sess)
            params.update(dict(sigma2inv_conc=sigma2inv_conc, sigma2inv_rate=sigma2inv_rate))

        return params






