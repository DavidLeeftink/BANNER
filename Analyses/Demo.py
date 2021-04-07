# %%
from Likelihoods import WishartProcessLikelihood
from Models import WishartProcess
import tensorflow as tf
import gpflow
from gpflow.utilities import print_summary
from gpflow.kernels import Sum, Cosine, SquaredExponential
import numpy as np
from numpy.random import uniform, normal
from scipy.stats import norm
np.random.seed(2022)
tf.random.set_seed(2021)
import matplotlib.pyplot as plt

### Generate synthetic data & visualize results
T = 10
N = 500  # time points
D = 4
true_mean = np.zeros(D)
true_cov = np.array([[.1,0,0.8,0],[0,.1,0,0.9],[0.8,0,.1,0],[0,0.9,0,.1]])

X = np.array([np.linspace(0, T, N) for i in range(D)]).T  # input time points
Y = np.random.multivariate_normal(true_mean, true_cov, N)
#Y = Y + 0.2*np.cos(0.6 * np.ones(D)) + 0.3*np.sin(1.1 * X *np.ones(D))
data = (X, Y)

fig, ax = plt.subplots(1, 1)
ax.imshow(true_cov, cmap='Reds')
ax.set_title(f'True covariance')
plt.show()

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

### Model initialization
# Model/training parameters
DoF = D  # Degrees of freedom
latent_dim = int(DoF * D)
R = 10  # samples for variational expectation
M = N  # num inducing point. exact (non-sparse) model is obtained by setting M=N
shared_kernel = True  # shares the same kernel parameters across input dimension if true

if M == N:
    Z_init = tf.identity(X)  # X.copy()
else:
    Z_init = np.array([np.linspace(0, T, M) for i in range(D)]).T  # .reshape(M,1) # initial inducing variable locations

max_iter = 2000

# Kernel
if shared_kernel:
    kernel = gpflow.kernels.SharedIndependent(
        Sum([SquaredExponential() * Cosine(lengthscales=8. / (i + 1)) for i in range(3)]), output_dim=latent_dim)
    kernel = gpflow.kernels.SharedIndependent(SquaredExponential(lengthscales=0.1),output_dim=latent_dim)
else:
    kern_list = [Sum([SquaredExponential() * Cosine(lengthscales=5. / (i + 1)) for i in range(2)]) for _ in range(latent_dim)]
    kernel = gpflow.kernels.SeparateIndependent(kern_list)

# Likelihood
model_inverse = False
additive_noise = True
likelihood = WishartProcessLikelihood.FullWishartLikelihood(D, DoF, R=R,
                                                            additive_noise=additive_noise,
                                                            model_inverse=model_inverse)


# Construct model
Z = tf.identity(Z_init)  # Z_init.copy()
iv = gpflow.inducing_variables.SharedIndependentInducingVariables(
    gpflow.inducing_variables.InducingPoints(Z))  # multi output inducing variables

wishart_process = WishartProcess.FullCovarianceWishartProcess(kernel, likelihood, D=D, DoF=DoF, inducing_variable=iv)
if M == N:
    gpflow.set_trainable(wishart_process.inducing_variable, False)
print_summary(wishart_process)


# train model
optimizer = gpflow.optimizers.Scipy()
optimizer.minimize(wishart_process.training_loss_closure(data),
                   variables=wishart_process.trainable_variables,
                   method="l-bfgs-b",
                   options={"disp": True, "maxiter": max_iter},)
print_summary(wishart_process)

# inspect resulting covariance matrix
print(f"ELBO: {wishart_process.elbo(data):.3}")
Sigma = wishart_process.mcmc_predict_density(X, Y, 100)
mean_Sigma = tf.reduce_mean(Sigma, axis=0)

# visualize latent functions for debugging
mu, var = wishart_process.predict_f(X)
fig, ax = plt.subplots(latent_dim,1, sharex=True, sharey=True, figsize=(10, 20))
colors = ['darkred', 'firebrick', 'red', 'salmon']
for i in range(latent_dim):
    # ax[i].plot(X[:, i%4], Y[:, i%4], color='black', label='Observations')
    # mean
    ax[i].plot(X[:, i%4], mu[:, i], color=colors[i%4], label='Posterior mean')
    ax[i].set_xlim((0, T))

    # 2*std
    top = mu[:, i] + 2.0 * var[:, i] ** 0.5
    bot = mu[:, i] - 2.0 * var[:, i] ** 0.5
    ax[i].fill_between(X[:, i%4], top, bot, alpha=0.3, color=colors[i%4], label='2$\sigma$')
    ax[i].set_xlim((0, T))
    if i == 2:
        ax[i].set_ylabel('measurement')
    if i == 3:
        ax[i].set_xlabel('time')
plt.show()


# Plot covariance matrix for first 10 timepoints
time_points = np.linspace(0,M,3)
for t, val in enumerate(time_points): #time_points:
    fig, ax = plt.subplots(1,1)
    ax.imshow(mean_Sigma[t], cmap='Reds')
    ax.set_title(f'time {round(val)}')
    plt.show()


