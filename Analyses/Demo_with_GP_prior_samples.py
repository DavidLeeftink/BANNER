# %%
from Likelihoods import WishartProcessLikelihood
from Models import WishartProcess
import tensorflow as tf
import gpflow
from gpflow.utilities import print_summary
from gpflow.kernels import Sum, Cosine, SquaredExponential, Periodic, Linear, SharedIndependent
from gpflow.inducing_variables import SharedIndependentInducingVariables, InducingPoints
from gpflow.ci_utils import ci_niter
import numpy as np
from numpy.random import uniform, normal
from scipy.stats import norm
np.random.seed(2022)
tf.random.set_seed(2021)
import matplotlib.pyplot as plt

################################################
#####  Create synthetic data from GP prior #####
################################################

## data properties
T = 10
N = 100
D = 2
X = np.array([np.linspace(0, T, N) for i in range(D)]).T  # input time points
true_lengthscale = 2.5

## Model properties
model_inverse = False
additive_noise = True
DoF = D+1  # Degrees of freedom
latent_dim = int(DoF * D)
R = 10  # samples for variational expectation
M = 10  # num inducing point. exact (non-sparse) model is obtained by setting M=N
shared_kernel = True  # shares the same kernel parameters across input dimension if true

if M == N:
    Z_init = tf.identity(X)  # X.copy()
else:
    Z_init = np.array([np.linspace(0, T, M) for i in range(D)]).T  # .reshape(M,1) # initial inducing variable locations
Z = tf.identity(Z_init)
iv = SharedIndependentInducingVariables(InducingPoints(Z))  # multi output inducing variables

## create GP model for the prior
kernel_prior = SharedIndependent(SquaredExponential(lengthscales=true_lengthscale),output_dim=latent_dim)
likelihood_prior = WishartProcessLikelihood.FullWishartLikelihood(D, DoF, R=R,
                                                            additive_noise=additive_noise,
                                                            model_inverse=model_inverse)
wishart_process_prior = WishartProcess.FullCovarianceWishartProcess(kernel_prior, likelihood_prior, D=D, DoF=DoF, inducing_variable=iv)
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

## create data by sampling from mvn at every timepoint
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
plt.show()


################################
#####  Generate GWP model  #####
################################

# Kernel
if shared_kernel:
    kernel = SharedIndependent(SquaredExponential(lengthscales=1.),output_dim=latent_dim)
else:
    kern_list = [Sum([SquaredExponential() * Cosine(lengthscales=2. / (i + 1)) for i in range(2)]) for _ in range(latent_dim)]
    kernel = gpflow.kernels.SeparateIndependent(kern_list)

# likelihood
likelihood = WishartProcessLikelihood.FullWishartLikelihood(D, DoF, R=R,
                                                            additive_noise=additive_noise,
                                                            model_inverse=model_inverse)
# create gwp model
wishart_process = WishartProcess.FullCovarianceWishartProcess(kernel, likelihood, D=D, DoF=DoF, inducing_variable=iv)
if M == N:
    gpflow.set_trainable(wishart_process.inducing_variable, False)
print_summary(wishart_process)

#################################
#####  Training & Inference #####
#################################

def run_adam(model, iterations, learning_rate=0.01, minibatch_size=25,  plot=False):
    """
    Utility function running the Adam optimizer.
    Note: optimization runs faster with auxilary function
    To do: put this to a util file

    :param model: GPflow model
    :param interations: number of iterations
    """
    # Create an Adam Optimizer action
    logf = []
    train_dataset = tf.data.Dataset.from_tensor_slices(data).repeat().shuffle(N)
    train_iter = iter(train_dataset.batch(minibatch_size))

    training_loss = model.training_loss_closure(train_iter, compile=True)
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    @tf.function
    def optimization_step():

        optimizer.minimize(training_loss, model.trainable_variables)
    n_steps_per_print = 100
    for step in range(iterations):
        optimization_step()
        if step % n_steps_per_print == 0:
            elbo = -training_loss().numpy()
            logf.append(elbo)
            print(f'Iteration {step}/{iterations}. ELBO: {elbo}')

    if plot:
        plt.figure()
        plt.plot(np.arange(max_iter)[::n_steps_per_print], logf)
        plt.xlabel("iteration")
        _ = plt.ylabel("ELBO")
        plt.title('Training convergence')
        plt.show()
    return logf

## optimization parameters
max_iter = ci_niter(20000)
learning_rate = 0.01
minibatch_size = 25

run_adam(wishart_process, max_iter, learning_rate, minibatch_size, plot=True)
print_summary(wishart_process)

# obtain output covariance matrix. To do: make a function in GWP class for this.
n_posterior_samples = 10000
print(f"ELBO: {wishart_process.elbo(data):.3}")
Sigma = wishart_process.mcmc_predict_density(X, Y, n_posterior_samples)
mean_Sigma = tf.reduce_mean(Sigma, axis=0)
var_Sigma = tf.math.reduce_variance(Sigma, axis=0)

##############################
#####  Visualize results #####
##############################

def plotMarginalCovariance(time, Sigma_mean, Sigma_var, Sigma_gt, samples=None):
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
                axes[i, j].fill_between(time[:,i], bot, top,color='red', alpha=0.05, zorder=-10, label='95% HDI')
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
    plt.suptitle('Marginal $\Sigma(t)$', fontsize=14)

plotMarginalCovariance(X, mean_Sigma, var_Sigma, Sigma_gt, samples=None)
plt.show()