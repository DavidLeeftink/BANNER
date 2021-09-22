# %%
from src.likelihoods import WishartProcessLikelihood
from src.models import WishartProcess
import tensorflow as tf
import gpflow
from gpflow.utilities import print_summary
from gpflow.kernels import Sum, Cosine, SquaredExponential
import numpy as np
from numpy.random import uniform, normal
np.random.seed(2021)
tf.random.set_seed(2021)
import matplotlib.pyplot as plt

### Generate synthetic data & visualize results
N, D = 250, 4
T = 10
X = np.array([np.linspace(0, T, N) for i in range(D)]).T
noise = normal(0, 1, (N, D))
Y = uniform(1.5, 2, D) * np.cos(0.6 * X + uniform(0, 2 * np.pi, D)) * np.sin(1.1 * X + uniform(0, 2 * np.pi, D)) + noise
Y[:,1:] = np.zeros((N,3))
# add connectivity between 1 and 4
for n in range(2,N):
    Y[n,1] += 1.*Y[n-2,0]
for n in range(12, N):
    Y[n,2] += 1. * Y[n-12, 0]
for n in range(22,N):
    Y[n,3] += 1.*Y[n-22,0]
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
    kernel = gpflow.kernels.SharedIndependent(SquaredExponential(lengthscales=1.),output_dim=latent_dim)
else:
    kern_list = [Sum([SquaredExponential() * Cosine(lengthscales=5. / (i + 1)) for i in range(2)]) for _ in range(D)]
    kernel = gpflow.kernels.SeparateIndependent(kern_list)

# Likelihood
model_inverse = False
additive_noise = False
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

# Plot covariance matrix for first 10 timepoints
time_points = np.arange(0,150,step=15)
for t in time_points:
    fig, ax = plt.subplots(1,1)
    ax.imshow(mean_Sigma[t], cmap='Reds')
    ax.set_title(f'time {t}')
    plt.show()


