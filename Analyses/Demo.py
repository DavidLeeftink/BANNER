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
import matplotlib.animation as animation
np.random.seed(2022)
tf.random.set_seed(2022)

#############################
#####  Model parameters #####
#############################
model_inverse = False
additive_noise = True
shared_kernel = True  # shares the same kernel parameters across input dimension
D = 3

DoF = D+1  # Degrees of freedom
n_inducing = 100  # num inducing point. exact (non-sparse) model is obtained by setting M=N
R = 10  # samples for variational expectation
latent_dim = int(DoF * D)


################################################
#####  Create synthetic data from GP prior #####
################################################

## data properties
T = 10
N = 250
X = np.array([np.linspace(0, T, N) for _ in range(D)]).T # input time points
true_lengthscale = 0.5

if n_inducing == N:
    Z_init = tf.identity(X)  # X.copy()
else:
    Z_init = np.array([np.linspace(0, T, n_inducing) for _ in range(D)]).T  # .reshape(M,1) # initial inducing variable locations
Z = tf.identity(Z_init)
iv = SharedIndependentInducingVariables(InducingPoints(Z))  # multi output inducing variables
#iv = InducingPoints(Z)

## create GP model for the prior

kernel_prior = SquaredExponential(lengthscales=true_lengthscale)

# i) all dims have the same kernel/lengthscale
kernel_prior = SharedIndependent(kernel_prior,output_dim=latent_dim)

# ii) all dims have a unique kernel/lengthscale
#kernel_prior = SeparateIndependent([SquaredExponential(lengthscales=true_lengthscale-1.8*(i==0)) for i in range(latent_dim)])

# iii) all vertices have a unique kernel/lengthscale
#kernel_prior = SharedIndependent(Sum([SquaredExponential(lengthscales=0.5+i/5, active_dims=np.arange(DoF)+i*DoF) for i in range(D)]), output_dim=latent_dim)

# iv) bare multi output kernel for testing kernel building
#kernel_prior = PartlySharedIndependentMOK([SquaredExponential(lengthscales=1.-0.7*(i==0)) for i in range(latent_dim)])
#kernel_prior = SeparateIndependent([SharedIndependent(SquaredExponential(lengthscales=1+i), output_dim=DoF) for i in range(D)])


likelihood_prior = WishartLikelihood(D, DoF, R=R, additive_noise=additive_noise, model_inverse=model_inverse)
#q_mu = tf.zeros([int(n_inducing*latent_dim), 1], dtype=gpflow.config.default_float())
#q_sqrt = tf.eye(int(n_inducing*latent_dim), dtype=gpflow.config.default_float())[tf.newaxis, ...]
wishart_process_prior = WishartProcess(kernel_prior, likelihood_prior, D=D, DoF=DoF, inducing_variable=iv)#, q_mu=q_mu, q_sqrt=q_sqrt)
print_summary(wishart_process_prior)

f_sample = wishart_process_prior.predict_f_samples(X, 1)
A = np.identity(D)
f_sample = tf.reshape(f_sample, [N, D, -1]) # (n_samples, D, DoF)
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


# # animation function.  This is called sequentially
# def animate(i):
#     im.set_array(Sigma_gt[i])
#     ax.set_title(f'X_n = {round((i / N) * T, 1)}')
#     return [im]
#
# fig, ax = plt.subplots()
# im = ax.imshow(Sigma_gt[0],cmap='Blues')
# fig.colorbar(im)
#
# anim = animation.FuncAnimation( fig, animate, frames = N,interval = 7000 / N)
# anim.save('covariance.mp4')
#
# print('Done!')
#
# assert 1==2


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

# Kernel
kernel = SquaredExponential(lengthscales=1.)

if shared_kernel:
    kernel = SharedIndependent(kernel, output_dim=latent_dim)
else:
    #kernel = SeparateIndependent([SquaredExponential(lengthscales=1.-(i+6)*0.01) for i in range(latent_dim)])
    #kernel = PartlySharedIndependentMOK([SquaredExponential(lengthscales=1.-(i+6)*0.01) for i in range(latent_dim)], DoF=DoF)
    kernel_list = [SharedIndependent([SquaredExponential(lengthscales=1+1*i) for i in range(DoF)], output_dim=DoF)
                    for _ in range(D)]
    kernel = CustomSeparateIndependent(kernel_list)

# likelihood
likelihood = WishartLikelihood(D, DoF, R=R, additive_noise=additive_noise, model_inverse=model_inverse)
# create gwp model
wishart_process = WishartProcess(kernel, likelihood, D=D, DoF=DoF, inducing_variable=iv)

# likelihood = FactorizedWishartLikelihood(D, DoF, n_factors=3, R=R, model_inverse=model_inverse)
# wishart_process = FactorizedWishartModel(kernel, likelihood, D=D, DoF=DoF, inducing_variable=iv)

if n_inducing == N:
    gpflow.set_trainable(wishart_process.inducing_variable, False)

print('wishart process model: (untrained)')
print_summary(wishart_process)

#################################
#####  Training & Inference #####
#################################

# optimization parameters
max_iter = ci_niter(15000)
learning_rate = 0.01
minibatch_size = 25

# train model, obtain output
run_adam(wishart_process, data, max_iter, learning_rate, minibatch_size, natgrads=False, plot=True)
print_summary(wishart_process)
print(f"ELBO: {wishart_process.elbo(data):.3}")

n_posterior_samples = 20000
Sigma = wishart_process.predict_mc(X, Y, n_posterior_samples)
mean_Sigma = tf.reduce_mean(Sigma, axis=0)
var_Sigma = tf.math.reduce_variance(Sigma, axis=0)

# np.save('X', X)
# np.save('Y', Y)
# np.save('mean_Sigma', mean_Sigma)
# np.save('var_Sigma', var_Sigma)
# np.save('gt_Sigma', Sigma_gt)
# np.save('10_posterior_samples_Sigma', Sigma[:10])
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
plt.show()
