from src.likelihoods.WishartProcessLikelihood import *
from src.likelihoods.FactorizedWishartLikelihood import *
from src.kernels.PartlySharedIndependentMOK import PartlySharedIndependentMultiOutput
from src.models.WishartProcess import *
from src.models.FactorizedWishartProcess import *
from util.training_util import *
import tensorflow as tf
import gpflow
from gpflow.utilities import print_summary
from gpflow.kernels import SquaredExponential
from gpflow.kernels import SharedIndependent
from gpflow.inducing_variables import SharedIndependentInducingVariables, InducingPoints
from gpflow.ci_utils import ci_niter
import numpy as np
from numpy.random import uniform
import matplotlib.pyplot as plt
np.random.seed(2023)
tf.random.set_seed(2023)

#############################
#####  Model parameters #####
#############################
model_inverse = False
additive_noise = True
multiple_observations = True
D = 7
n_factors = 5

nu = n_factors + 1
N = 100
n_inducing = 50  # num inducing point. exact (non-sparse) model is obtained by setting M=N
R = 5  # samples for variational expectation
latent_dim = int(nu * D)

# optimization parameters
max_iter = ci_niter(10000)
learning_rate = 0.01
minibatch_size = 25

# Kernel
# kernel = SquaredExponential(lengthscales=1.)
# kernel = SharedIndependent(kernel, output_dim=latent_dim)
kernel = PartlySharedIndependentMultiOutput([SquaredExponential(lengthscales=0.3 + i*0.1) for i in range(n_factors)], nu=nu)

################################################
#####  Create synthetic data from GP prior #####
################################################

## data properties
T = 10
time_window = 5
X = np.array([np.linspace(0, time_window, N) for _ in range(D)]).T  # input time points
true_lengthscale = 1.4

if n_inducing == N:
    Z_init = tf.identity(X)  # X.copy()
else:
    Z_init = np.array([np.linspace(0, time_window, n_inducing) for _ in
                       range(D)]).T  # .reshape(M,1) # initial inducing variable locations
Z = tf.identity(Z_init)
iv = SharedIndependentInducingVariables(InducingPoints(Z))  # multi output inducing variables

## create GP model for the prior
squared_exponential = SquaredExponential(lengthscales=true_lengthscale)
kernel_prior = SharedIndependent(squared_exponential, output_dim=latent_dim)
likelihood_prior = WishartLikelihood(D, nu, R=R, additive_noise=additive_noise, model_inverse=model_inverse)
wishart_process_prior = WishartProcess(kernel_prior, likelihood_prior, D=D, nu=nu,
                                       inducing_variable=iv)  # , q_mu=q_mu, q_sqrt=q_sqrt)
print('wishart process model: (prior)')
print_summary(wishart_process_prior)

# Sample true function
f_sample = wishart_process_prior.predict_f_samples(X, 1)
A = np.identity(D)
f_sample = tf.reshape(f_sample, [N, D, -1])  # (n_samples, D, nu)
Sigma_gt = np.matmul(f_sample, np.transpose(f_sample, [0, 2, 1]))

# fig, ax = plt.subplots(D, D, figsize=(10, 10))
# for i in range(D):
#     for j in range(D):
#         if i <= j:
#             ax[i, j].set_title(r'$\Sigma_{{{:d}{:d}}}$'.format(i, j))
#             ax[i, j].plot(X, Sigma_gt[:, i, j], color='C0', label='True function')
#         else:
#             ax[i, j].axis('off')
# plt.show()

# create data by sampling from mvn at every timepoint
if multiple_observations:
    Y = np.zeros((N, T, D))
    for n in range(N):
        for t in range(T):
            Y[n, t, :] = np.random.multivariate_normal(mean=np.zeros((D)), cov=Sigma_gt[n, :, :])
else:
    Y = np.zeros((N, D))
    for n in range(N):
        Y[n, :] = np.random.multivariate_normal(mean=np.zeros((D)), cov=Sigma_gt[n, :, :])
data = (X, Y)

################################
#####  Generate GWP model  #####
################################

# Factorized model
likelihood = FactorizedWishartLikelihood(D, nu, n_factors=n_factors, R=R,
                                         model_inverse=model_inverse, multiple_observations=multiple_observations)
wishart_process = FactorizedWishartModel(kernel, likelihood, D=n_factors, nu=nu, inducing_variable=iv, num_data=X.shape[0])

# Non-factorized model
# likelihood = WishartLikelihood(D, nu, R=R, additive_noise=additive_noise, model_inverse=model_inverse,
#                                multiple_observations=multiple_observations)
# wishart_process = WishartProcess(kernel, likelihood, D=D, nu=nu, inducing_variable=iv)

if n_inducing == N:
    gpflow.set_trainable(wishart_process.inducing_variable, False)

print('wishart process model: (untrained)')
print_summary(wishart_process)

#################################
#####  Training & Inference #####
#################################

# train model, obtain output
run_adam(wishart_process, data, max_iter, learning_rate, minibatch_size, natgrads=False)
print_summary(wishart_process)
print(f"ELBO: {wishart_process.elbo(data):.3}")

n_posterior_samples = 5000
Sigma = wishart_process.predict_mc(X, n_posterior_samples)

mean_Sigma = tf.reduce_mean(Sigma, axis=0)
var_Sigma = tf.math.reduce_variance(Sigma, axis=0)

##############################
#####  Visualize results #####
##############################

def plot_marginal_covariance(time, Sigma_mean, Sigma_var, Sigma_gt, samples=None):
    N, _, D = Sigma_gt.shape

    f, axes = plt.subplots(nrows=D, ncols=D, figsize=(20, 20), sharey=True, sharex=True)
    for i in range(D):
        for j in range(D):
            if i <= j:
                axes[i, j].plot(time, Sigma_gt[:, i, j], label='Ground truth', color='C0')
                axes[i, j].plot(time, Sigma_mean[:, i, j], label='VB', zorder=-5, color='red')
                # 2 standard deviations from the mean =\approx 95%
                top = Sigma_mean[:, i, j] + 2.0 * Sigma_var[:, i, j] ** 0.5
                bot = Sigma_mean[:, i, j] - 2.0 * Sigma_var[:, i, j] ** 0.5
                # plot std -> to do
                axes[i, j].fill_between(time[:, i], bot, top, color='red', alpha=0.05, zorder=-10, label='95% HDI')
                if samples is not None:
                    for m in range(samples.shape[0]):
                        axes[i, j].plot(time, samples[m, :, i, j], label='function samples', zorder=-5, color='red', alpha=0.15)
                if i == j:
                    axes[i, j].set_title('Marginal variance {:d}'.format(i))
                else:
                    axes[i, j].set_title(r'Marginal covariance $\Sigma_{{{:d}{:d}}}$'.format(i, j))
                axes[i, j].set_xlabel('Time')
                # if i == D - 1 and j == D - 1:
                #     axes[i, j].legend()
            else:
                axes[i, j].axis('off')

    plt.subplots_adjust(top=0.9)
    plt.suptitle('BANNER: Marginal $\Sigma(t)$', fontsize=14)


plot_marginal_covariance(X, mean_Sigma, var_Sigma, Sigma_gt, samples=Sigma[:5])
plt.figure()
plt.show()
