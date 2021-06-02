from Likelihoods.WishartProcessLikelihood import *
from Models.WishartProcess import *
from Models.training_util import *
from Kernels.PartlySharedIndependentMOK import CustomSeparateIndependent
import tensorflow as tf
import gpflow
from gpflow.utilities import print_summary
from gpflow.kernels import Sum, Cosine, SquaredExponential, Periodic, Linear
from gpflow.kernels import SharedIndependent, SeparateIndependent
from gpflow.inducing_variables import SharedIndependentInducingVariables, InducingPoints
from gpflow.ci_utils import ci_niter
import numpy as np
from numpy.random import uniform, normal
import matplotlib.pyplot as plt
np.random.seed(2022)
tf.random.set_seed(2022)

#############################
#####  Model parameters #####
#############################
model_inverse = False
additive_noise = True
shared_kernel = True  # shares the same kernel parameters across input dimension
D = 4
n_factors = 3

nu = n_factors + 1  # Degrees of freedom
n_inducing = 100  # num inducing point. exact (non-sparse) model is obtained by setting M=N
R = 10  # samples for variational expectation
latent_dim = int(nu * D)

# Kernel
kernel = SquaredExponential(lengthscales=1.)
if shared_kernel:
    kernel = SharedIndependent(kernel, output_dim=latent_dim)
else:
    kernel = SeparateIndependent([SquaredExponential(lengthscales=1.-(i+6)*0.01) for i in range(latent_dim)])

################################################
#####  Create synthetic data from GP prior #####
################################################

## data properties
T = 5
N = 100
X = np.array([np.linspace(0, T, N) for _ in range(D)]).T # input time points
true_lengthscale = 0.5

if n_inducing == N:
    Z_init = tf.identity(X)  # X.copy()
else:
    Z_init = np.array([np.linspace(0, T, n_inducing) for _ in range(D)]).T  # .reshape(M,1) # initial inducing variable locations
Z = tf.identity(Z_init)
iv = SharedIndependentInducingVariables(InducingPoints(Z))  # multi output inducing variables

## create GP model for the prior
squared_exponential = SquaredExponential(lengthscales=true_lengthscale)
kernel_prior = SharedIndependent(squared_exponential,output_dim=latent_dim)
likelihood_prior = WishartLikelihood(D, nu, R=R, additive_noise=additive_noise, model_inverse=model_inverse)
wishart_process_prior = WishartProcess(kernel_prior, likelihood_prior, D=D, nu=nu, inducing_variable=iv)#, q_mu=q_mu, q_sqrt=q_sqrt)
print('wishart process model: (prior)')
print_summary(wishart_process_prior)

# Sample true function
f_sample = wishart_process_prior.predict_f_samples(X, 1)
A = np.identity(D)
f_sample = tf.reshape(f_sample, [N, D, -1]) # (n_samples, D, nu)
Sigma_gt = np.matmul(f_sample, np.transpose(f_sample, [0, 2, 1]))

fig, ax = plt.subplots(D,D,figsize=(10,10))
for i in range(D):
    for j in range(D):
        if i <= j:
            ax[i,j].set_title(r'$\Sigma_{{{:d}{:d}}}$'.format(i, j))
            ax[i,j].plot(X, Sigma_gt[:,i,j], color='C0',label='True function')
        else:
            ax[i, j].axis('off')
plt.show()

# create data by sampling from mvn at every timepoint
Y = np.zeros((N, D))
for t in range(N):
    Y[t, :] = np.random.multivariate_normal(mean=np.zeros((D)), cov=Sigma_gt[t, :, :])
data = (X, Y)

fig, ax = plt.subplots(D, 1, sharex=True, sharey=True, figsize=(10, 6))
if not isinstance(ax, np.ndarray):
    ax = [ax]
colors = ['darkred', 'firebrick', 'red', 'salmon']
for i in range(D):
    ax[i].plot(X[:, i], Y[:, i], color=colors[i])
    ax[i].set_xlim((0, T))
    if i == 2:
        ax[i].set_ylabel('measurement')
    if i == 3:
        ax[i].set_xlabel('time')

#plt.savefig('data.png')
plt.show()

################################
#####  Generate GWP model  #####
################################

likelihood = FactorizedWishartLikelihood(D, nu, n_factors=n_factors, R=R, model_inverse=model_inverse)
wishart_process = FactorizedWishartModel(kernel, likelihood, D=D, nu=nu, inducing_variable=iv)
# todo: should wishart_process be given n_factors instead of D? Since there are only n_factor x nu independent GPs?
if n_inducing == N:
    gpflow.set_trainable(wishart_process.inducing_variable, False)

print('wishart process model: (untrained)')
print_summary(wishart_process)

#################################
#####  Training & Inference #####
#################################

# optimization parameters
max_iter = ci_niter(1000)
learning_rate = 0.01
minibatch_size = 25

# train model, obtain output
run_adam(wishart_process, data, max_iter, learning_rate, minibatch_size, natgrads=False, plot=True)
print_summary(wishart_process)
print(f"ELBO: {wishart_process.elbo(data):.3}")

n_posterior_samples = 2000
Sigma = wishart_process.predict_mc(X, Y, n_posterior_samples)
mean_Sigma = tf.reduce_mean(Sigma, axis=0)
var_Sigma = tf.math.reduce_variance(Sigma, axis=0)
print(f'Mean Sigma shape: {mean_Sigma.shape}, var_Sigma shape: {var_Sigma.shape}')
##############################
#####  Visualize results #####
##############################

def plot_marginal_covariance(time, Sigma_mean, Sigma_var, Sigma_gt, samples=None):
    N, _, D = Sigma_gt.shape

    f, axes = plt.subplots(nrows=D, ncols=D, figsize=(12, 12))
    for i in range(D):
        for j in range(D):
            if i <= j:
                axes[i, j].plot(time, Sigma_gt[:, i, j], label='Ground truth', color='C0')
                axes[i, j].plot(time, Sigma_mean[:, i, j], label='VB', zorder=-5, color='red')
                # 2 standard deviations from the mean =\approx 95%
                top = Sigma_mean[:, i, j] + 2.0 * Sigma_var[:, i, j] ** 0.5
                bot = Sigma_mean[:, i, j] - 2.0 * Sigma_var[:, i, j] ** 0.5
                # plot std -> to do
                axes[i, j].fill_between(time[:,i], bot, top, color='red', alpha=0.05, zorder=-10, label='95% HDI')
                if samples is not None:
                    axes[i, j].plot(time, samples[:, i, j], label='function samples', zorder=-5, color='red', alpha=0.2)
                if i == j:
                    axes[i, j].set_title('Marginal variance {:d}'.format(i))
                else:
                    axes[i, j].set_title(r'Marginal covariance $\Sigma_{{{:d}{:d}}}$'.format(i, j))
                axes[i, j].set_xlabel('Time')
                if i == D - 1 and j == D - 1:
                    axes[i, j].legend()
            else:
               axes[i, j].axis('off')

    plt.subplots_adjust(top=0.9)
    plt.suptitle('BANNER: Marginal $\Sigma(t)$', fontsize=14)

plot_marginal_covariance(X, mean_Sigma, var_Sigma, Sigma_gt, samples=None)
plt.figure()
plt.show()