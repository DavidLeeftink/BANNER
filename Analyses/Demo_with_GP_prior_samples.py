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


## Generate synthetic data & visualize results
T = 10
N = 100  # time points
D = 2
X = np.array([np.linspace(0, T, N) for i in range(D)]).T  # input time points
true_lengthscale = 2.5

## sample data from prior of GP model
model_inverse = False
additive_noise = True # important -> unstable if false (without white noise)
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

kernel_prior = SharedIndependent(SquaredExponential(lengthscales=true_lengthscale),output_dim=latent_dim)
likelihood_prior = WishartProcessLikelihood.FullWishartLikelihood(D, DoF, R=R,
                                                            additive_noise=additive_noise,
                                                            model_inverse=model_inverse)
wishart_process_prior = WishartProcess.FullCovarianceWishartProcess(kernel_prior, likelihood_prior, D=D, DoF=DoF, inducing_variable=iv)
f_sample = wishart_process_prior.predict_f_samples(X, 1)
A = np.identity(D)
f_sample = tf.reshape(f_sample, [N, D, -1]) # (n_samples, D, DoF)
affa = np.matmul(f_sample, np.transpose(f_sample, [0, 2, 1]))

fig, ax = plt.subplots(D,D,figsize=(10,10))
for i in range(D):
    for j in range(D):
        ax[i,j].set_title(f'Sigma {i}{j}')
        ax[i,j].plot(X, affa[:,i,j], color='C0',label='True function')
plt.show()


## create data by sampling from mvn at every timepoint
Y = np.zeros((N, D))
for t in range(N):
    Y[t, :] = np.random.multivariate_normal(mean=np.zeros((D)), cov=affa[t, :, :])

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


# np.save('testdata_covariance', affa)
# np.save('testdata_X', X)
# np.save('testdata_Y', Y)


### Model initialization

# Kernel
if shared_kernel:
    kernel = SharedIndependent(SquaredExponential(lengthscales=10.,variance=1.),output_dim=latent_dim)
else:
    kern_list = [Sum([SquaredExponential() * Cosine(lengthscales=2. / (i + 1)) for i in range(2)]) for _ in range(latent_dim)]
    kernel = gpflow.kernels.SeparateIndependent(kern_list)

# Construct likelihood and model
likelihood = WishartProcessLikelihood.FullWishartLikelihood(D, DoF, R=R,
                                                            additive_noise=additive_noise,
                                                            model_inverse=model_inverse)
wishart_process = WishartProcess.FullCovarianceWishartProcess(kernel, likelihood, D=D, DoF=DoF, inducing_variable=iv)

if M == N:
    gpflow.set_trainable(wishart_process.inducing_variable, False)

# train model
max_iter = ci_niter(10000)
learning_rate = 0.01
print(wishart_process.trainable_variables)


def run_adam(model, iterations, learning_rate):
    """
    Utility function running the Adam optimizer

    :param model: GPflow model
    :param interations: number of iterations
    """
    # Create an Adam Optimizer action
    logf = []
    minibatch_size = 25
    train_dataset = tf.data.Dataset.from_tensor_slices((X, Y)).repeat().shuffle(N)
    train_iter = iter(train_dataset.batch(minibatch_size))

    training_loss = model.training_loss_closure(train_iter, compile=True)
    optimizer = tf.optimizers.Adam()

    @tf.function
    def optimization_step():

        optimizer.minimize(training_loss, model.trainable_variables)

    for step in range(iterations):
        optimization_step()
        if step % 10 == 0:
            elbo = -training_loss().numpy()
            logf.append(elbo)
    return logf

logf = run_adam(wishart_process, max_iter, learning_rate)
plt.figure()
plt.plot(np.arange(max_iter)[::10], logf)
plt.xlabel("iteration")
_ = plt.ylabel("ELBO")
plt.show()
print_summary(wishart_process)

# obtain output covariance matrix
n_posterior_samples = 10000
print(f"ELBO: {wishart_process.elbo(data):.3}")
Sigma = wishart_process.mcmc_predict_density(X, Y, n_posterior_samples)
mean_Sigma = tf.reduce_mean(Sigma, axis=0)
var_Sigma = tf.math.reduce_variance(Sigma, axis=0)
print(mean_Sigma.shape)

## visualize results
fig, ax = plt.subplots(D,D,figsize=(10,10))
for i in range(D):
    for j in range(D):
        ax[i,j].set_title(f'Sigma {i}{j}')
        top = mean_Sigma[:,i,j] + 2.0 * var_Sigma[:,i,j] ** 0.5
        bot = mean_Sigma[:,i,j] - 2.0 * var_Sigma[:,i,j] ** 0.5
        if i==j:
            ax[i,j].plot(X, affa[:,i,j], color='C0',label='True function')
            ax[i,j].plot(X, mean_Sigma[:,i,j], color='red', label='Posterior expectation')
            ax[i,j].fill_between(X[:,i], top, bot, alpha=0.3, color='red', label='2$\sigma$')
        else:
            ax[i, j].plot(X, affa[:, i, j], color='C0')
            ax[i, j].plot(X, mean_Sigma[:, i, j], color='red')
            ax[i,j].fill_between(X[:,i], top, bot, alpha=0.3, color='red')
ax[i,j].legend()
plt.show()