import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import gpflow
from gpflow.kernels import SquaredExponential, SharedIndependent, SeparateIndependent
from gpflow.inducing_variables import SharedIndependentInducingVariables, InducingPoints
from gpflow.utilities import print_summary
from gpflow.ci_utils import ci_niter

from Models.WishartProcess import WishartProcess
from Models.training_util import run_adam
from Kernels.PartlySharedIndependentMOK import CustomMultiOutput
from Likelihoods.WishartProcessLikelihood import WishartLikelihood

# mpl.rcParams['pdf.fonttype'] = 42
# mpl.rcParams['ps.fonttype'] = 42
# # plt.rc('text', usetex=True)
# plt.rcParams['svg.fonttype'] = 'none'

plt.rc('axes', titlesize=24)  # fontsize of the axes title
plt.rc('axes', labelsize=24)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=18)  # fontsize of the tick labels
plt.rc('ytick', labelsize=18)  # fontsize of the tick labels
plt.rc('legend', fontsize=20)  # legend fontsize
plt.rc('figure', titlesize=20)  # fontsize of the figure title


###################################################################


def simulate_wishart_process(D=5, T=10, N=100, A=None, nu=None):
    """
    :param D: Number of observed variables.
    :param T: Length of time series.
    :param N: Number of samples.
    :param A: Cholesky decomposition of the Wishart scaling matrix, i.e. V=AA^T. Defaults to the identity.
    :param nu: The Wishart degrees of freedom. Defaults to D+1.
    :return: A sample from the Wishart process \Sigma(x)
    """

    def sample_gp(X, kernel, nsamples):
        """
        :param X: Input locations.
        :param kernel: GP kernel.
        :param nsamples: Number of samples.
        :return: Generates samples from a GP with known kernel.
        """
        K = kernel.K(X) + 1e-6 * np.eye(N)
        L = np.linalg.cholesky(K)
        z = np.random.normal(size=(N, nsamples))
        return np.dot(L, z)  # N x nsamples

    #
    if A is None:
        A = np.eye(D)

    # Number of latent GPs (aka degrees-of-freedom)
    if nu is None:
        nu = D + 1

    # Generate input
    X = np.tile(np.linspace(0, T, N), (D, 1)).T

    # True dynamic properties of each of D variables. We pick some random lengthscales and variances within some bounds.
    true_lengthscale = np.random.permutation(np.linspace(0.05 * T, 0.3 * T, num=D))
    true_variance = 1 + 0.5*np.random.uniform(size=D)

    # The true kernel is a distinct kernel for each latent GP. Across the degrees of freedom (\nu), the same kernel is
    # used.
    kernels = [SquaredExponential(lengthscales=true_lengthscale[i], variance=true_variance[i]) for i in range(D)]

    # Latent GPs
    u = np.zeros((D, nu, N))
    for i in range(D):
        u[i, :, :] = sample_gp(X, kernels[i], nsamples=nu).T

    Sigma = np.zeros((N, D, D))
    for t in range(N):
        Sigma[t, :, :] = np.dot(np.dot(A, np.dot(u[:, :, t], u[:, :, t].T)), A.T)

    return X, Sigma, true_lengthscale, true_variance


#
def sample_observations(Sigma, mu=None):
    """
    :param Sigma: Generalised Wishart Process, i.e. covariance process
    :param mu: Mean function.
    :return: Returns N x D matrix Y, containing one observation for each 'time point'.
    """

    N, D, _ = Sigma.shape
    if mu is None:
        mu = np.zeros(D)

    Y = np.zeros((N, D))
    for t in range(N):
        Y[t, :] = np.random.multivariate_normal(mean=mu, cov=Sigma[t, :, :])
    return Y


#
def run_wishart_process_inference(data, T, iterations=10000, num_inducing=None, learning_rate=0.01, batch_size=25):
    """

    :param data: Tuple (X, Y) of input and responses.
    :param T: Last timepoint (assume we start at t=0).
    :param iterations: Number of variational inference optimization iterations.
    :param num_inducing: Number of inducing points (inducing points are in the same space as X).
    :param learning_rate: Optimization parameter.
    :param batch_size: Data is split into batches of size batch_size, for stochastic optimization (otherwise we'll run
    out of memory).
    :return: Returns a dictionary with the posterior Wishart process (with trained hyperparameters and variational
    parameters), and a list with the ELBO (loss) per iteration.
    """
    X, Y = data
    N, D = Y.shape
    if num_inducing is None:
        num_inducing = int(0.4 * N)

    model_inverse = False
    additive_noise = True

    nu = D + 1  # Degrees of freedom
    R = 10  # samples for variational expectation
    latent_dim = int(nu * D)

    if num_inducing == N:
        Z_init = tf.identity(X)  # X.copy()
    else:
        Z_init = np.array([np.linspace(0, T, num_inducing) for _ in range(D)]).T   # initial inducing variable locations
    Z = tf.identity(Z_init)
    iv = SharedIndependentInducingVariables(InducingPoints(Z))  # multi output inducing variables

    kernel_type = 'partially_shared'  # ['shared', 'separate', 'partially_shared']   # shares the same kernel parameters across input dimension
    kernel = SquaredExponential(lengthscales=5.)

    if kernel_type == 'shared':
        kernel = SharedIndependent(kernel, output_dim=latent_dim)
    elif kernel_type == 'separate':
        kernel = SeparateIndependent([SquaredExponential(lengthscales=1. - (i + 6) * 0.01) for i in range(latent_dim)])
    elif kernel_type == 'partially_shared':
        kernel = CustomMultiOutput([SquaredExponential(lengthscales=0.5 + i * 0.5) for i in range(D)], nu=nu)
    else:
        raise NotImplementedError

    # likelihood
    likelihood = WishartLikelihood(D, nu, R=R, additive_noise=additive_noise, model_inverse=model_inverse)
    # create GWP model
    wishart_process = WishartProcess(kernel, likelihood, D=D, nu=nu, inducing_variable=iv)

    # If num_inducing==N, we do not actually have inducing points.
    if num_inducing == N:
        gpflow.set_trainable(wishart_process.inducing_variable, False)

    elbo = run_adam(wishart_process, data, ci_niter(iterations), learning_rate, batch_size, natgrads=False, pb=True)
    return {'wishart process': wishart_process, 'ELBO': elbo}


#
def plot_loss(num_iter, loss):
    x = np.linspace(1, num_iter, len(loss))
    fig = plt.figure()
    plt.plot(x, loss, label='ELBO')
    plt.xlim([x[0], x[-1]])
    plt.xlabel('Iteration')
    plt.ylabel('ELBO')
    plt.title('Training convergence')
    plt.tight_layout()


#
def plot_wishart_process_sample(X, Sigma):
    N, D, _ = Sigma.shape
    c1 = '#363537'
    c2 = '#EF2D56'
    fig, axes = plt.subplots(nrows=D, ncols=D, sharex=True, sharey=True, figsize=(10, 10))
    x = X[:, 0]

    for i in range(D):
        for j in range(D):
            ax = axes[i, j]
            if j < i:
                ax.axis('off')
            else:
                ax.plot(x, Sigma[:, i, j], color=c1, lw=2, label='Ground truth')
                ax.set_xlim([x[0], x[-1]])
                ax.set_title('$\sigma_{{{:d},{:d}}}(t)$'.format(i + 1, j + 1), fontsize=20)
    return fig, axes


#
def plot_wishart_predictions(samples, X, axes=None, plot_individuals=0):
    posterior_expectation = tf.reduce_mean(samples, axis=0).numpy()
    posterior_variance = tf.math.reduce_variance(samples, axis=0).numpy()

    c1 = '#363537'
    c2 = '#EF2D56'

    _, D = X.shape
    x = X[:, 0]
    if axes is None:
        fig, axes = plt.subplots(nrows=D, ncols=D, sharex=True, sharey=True, figsize=(10, 10))

    for i in range(D):
        for j in range(D):
            ax = axes[i, j]
            if j < i:
                ax.axis('off')
            else:
                mean = posterior_expectation[:, i, j]
                intv = 1.96*np.sqrt(posterior_variance[:, i, j])
                ax.plot(x, mean, lw=2, c=c2, label='BANNER posterior mean')
                ax.fill_between(x, mean - intv, mean + intv, color=c2, alpha=0.2, label='BANNER 95\% HDI')
                if plot_individuals > 0:
                    ixs = np.random.randint(0, samples.shape[0], size=plot_individuals)
                    for ix in ixs:
                        ax.plot(x, samples[ix, :, i, j].numpy(), c=c2, alpha=0.4, lw=0.5)
                ax.set_xlim([x[0], x[-1]])


#
def plot_sliding_window(V_mean, V_var, axes=None):

    c1 = '#363537'
    c2 = '#EF2D56'
    c3 = '#00563E'

    _, D = X.shape
    x = X[:, 0]
    if axes is None:
        fig, axes = plt.subplots(nrows=D, ncols=D, sharex=True, sharey=True, figsize=(10, 10))

    for i in range(D):
        for j in range(D):
            ax = axes[i, j]
            if j < i:
                ax.axis('off')
            else:
                mean = V_mean[:, i, j]
                intv = 1.96*np.sqrt(V_var[:, i, j])
                ax.plot(x, mean, lw=2, c=c3, label='Sliding-window posterior mean')
                ax.fill_between(x, mean - intv, mean + intv, color=c3, alpha=0.2, label='Sliding-window 95\% HDI')
                ax.set_xlim([x[0], x[-1]])
                # ax.set_title('$\sigma_{{{:d},{:d}}}(t)$'.format(i + 1, j + 1), fontsize=20)


#
def run_sliding_window(Y, window_size, stride_length=1):
    """

    :param Y: Responses Y, of the form N x D.
    :param window_size: Number of observations to put into one window.
    :param stride_length: How much we move for the next window (defaults to 1, i.e. no striding).
    :return: Returns a tuple, containing the sliding-window expectation of the covariance matrix, and the corresponding
    variance of the covariance matrix. Note: the latter is already based on a Bayesian sliding window that uses the
    inverse Wishart distribution (distribution, not process). Standard SW approaches do not have a natural way to
    quantify uncertainty.
    """
    def inverse_wishart_mean_and_variance(V, m):
        """

        :param V: Inverse Wishart scale matrix.
        :param m: Inverse Wishart degrees-of-freedom.
        :return: Returns expectation and variance of the inverse Wishart distribution. Because of conjugacy we can do
        this analytically; see e.g. https://isdsa.org/_media/jbds/v1n2/v1n2p2.pdf.
        """
        p = V.shape[0]
        inv_wish_expectation = 1 / (m - p - 1) * V
        a = np.tile(np.diag(V), (p, 1))
        aat = np.multiply(a, a.T)
        inv_wish_variance = ((m - p + 1) * V**2 + (m - p - 1)*aat) / ((m - p) * (m - p - 1) ** 2 * (m - p - 3))
        return inv_wish_expectation, inv_wish_variance

    #
    inv_wish_scale = np.eye(D)
    inv_wish_dof = D + 1

    InvSigma_sw_mean = np.zeros((N, D, D))
    InvSigma_sw_variance = np.zeros((N, D, D))
    for i in np.arange(0, N, step=stride_length):
        window_min, window_max = int(np.max([i - 0.5 * window_size, 0])), \
                                 int(np.min([i + 0.5 * window_size, N]))
        Y_window = Y[window_min:window_max, :]
        n_window = window_max - window_min
        S_window = 1 / (window_max - window_min) * np.dot(Y_window.T, Y_window)

        inv_wish_mean, inv_wish_var = inverse_wishart_mean_and_variance(S_window * n_window + inv_wish_scale,
                                                                        n_window + inv_wish_dof)

        InvSigma_sw_mean[i, :, :] = inv_wish_mean
        InvSigma_sw_variance[i, :, :] = inv_wish_var

    return InvSigma_sw_mean, InvSigma_sw_variance


#

#########################################################################

# D := number of variables
# N := number of observations
# T := maximum value, i.e. X = np.linspace(0, T, num=N)

# Note: For this simulation, we assume evenly spaced inputs; for BANNER this is not required, but sliding-window
# approaches need some interpolation here.

# Note: Individual samples from the GWP are very jittery; is this due to the additive white noise model, or due to the
# lack of posterior correlations in the GP samples?

# Note: Proper window length for sliding-window approach should be determined via cross-validation. Too small: variance
# becomes impractical. Too large: too smooth signal. Length-scale optimization has a similar issue!

D, N, T = 3, 100, 4.0
X, Sigma_true, true_lengthscale, true_variance = simulate_wishart_process(D=D, T=T, N=N)
Y = sample_observations(Sigma_true)

num_iter = 10000
gwp_results = run_wishart_process_inference(data=(X, Y), T=T,
                                            iterations=num_iter,
                                            num_inducing=int(0.4*N),
                                            batch_size=100)
posterior_wishart_process = gwp_results['wishart process']
elbo = gwp_results['ELBO']

print_summary(posterior_wishart_process)

plot_loss(num_iter, elbo)

window_size=10
stride_length=1

sliding_window_inv_wishart_mean, sliding_window_inv_wishart_var = run_sliding_window(Y,
                                                                                     window_size=window_size,
                                                                                     stride_length=stride_length)

num_samples = 5000

fig, axes = plot_wishart_process_sample(X, Sigma_true)
samples = posterior_wishart_process.predict_mc(X, num_samples)
plot_wishart_predictions(samples=samples, X=X, axes=axes, plot_individuals=5)
plot_sliding_window(sliding_window_inv_wishart_mean, sliding_window_inv_wishart_var, axes=axes)

axes[-1, -1].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('(Co)variance')
handles, labels = axes[-1, -1].get_legend_handles_labels()
plt.figlegend(handles, labels,
              ncol=1, loc='lower left', bbox_to_anchor=(0.1, 0.10), frameon=False)
plt.tight_layout()
plt.show()

for i in range(D):
    print('Variable {:d}'.format(i))
    print('True lengthscale: {:0.3f}'.format(true_lengthscale[i]))
    print('Estimated lengthscale: {:0.3f}'.format(posterior_wishart_process.kernel.kernels[i].kernel.lengthscales.numpy().item()))
    print('True variance: {:0.3f}'.format(true_variance[i]))
    print('Estimated variance: {:0.3f}'.format(posterior_wishart_process.kernel.kernels[i].kernel.variance.numpy().item()))
