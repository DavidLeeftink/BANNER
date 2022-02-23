import numpy as np
import matplotlib.pyplot as plt
import gpflow
from gpflow.kernels import SquaredExponential, Linear, SharedIndependent, SeparateIndependent
from gpflow.inducing_variables import SharedIndependentInducingVariables, InducingPoints
from gpflow.ci_utils import ci_niter
from src.kernels.PartlySharedIndependentMOK import PartlySharedIndependentMultiOutput
import time

gpflow.config.set_default_float(np.float64)
np.random.seed(0)
MAXITER = ci_niter(10000)

N = 100  # number of points
D = 1  # number of input dimensions
n_kernels, nu = 2, 2
M = 15  # number of inducing points
L = P = 4  # number of latent GPs, number of observations = output dimension

def generate_data(N=100):
    X = np.random.rand(N)[:, None] * 10 - 5  # Inputs = N x D
    G = np.hstack((0.5 * np.sin(X/2) + X, 3.0 * np.cos(X/2) - X,0.5 * np.sin(4 * X) + X, 3.0 * np.cos(4*X) - X))  # G = N x L
    W = np.array([[0.5, -0.3, 0, 0], [0.5, -0.3, 0, 0], [0, 0, -0.4, 0.6],[0.0, 0.0, 0.6, -0.4]])  # L x P
    F = np.matmul(G, W)  # N x P
    Y = F + np.random.randn(*F.shape) * [0.2, 0.2, 0.2, 0.2]
    return X, Y

X, Y = data = generate_data(N)
Zinit = np.linspace(-5, 5, M)[:, None]

def plot_model(m, name, lower=-7.0, upper=7.0):
    pX = np.linspace(lower, upper, 100)[:, None]
    pY, pYv = m.predict_y(pX)
    if pY.ndim == 3:
        pY = pY[:, 0, :]
    plt.plot(X, Y, "x")
    plt.gca().set_prop_cycle(None)
    plt.plot(pX, pY)
    for i in range(pY.shape[1]):
        top = pY[:, i] + 2.0 * pYv[:, i] ** 0.5
        bot = pY[:, i] - 2.0 * pYv[:, i] ** 0.5
        plt.fill_between(pX[:, 0], top, bot, alpha=0.3)
    plt.xlabel("X")
    plt.ylabel("f")
    plt.title(f"{name} kernel.")
    plt.show()

# initialization of inducing input locations (M random points from the training inputs)
Z = Zinit.copy()
# create multi-output inducing variables from Z
iv = SharedIndependentInducingVariables(InducingPoints(Z))

def optimize_model_with_scipy(model):
    optimizer = gpflow.optimizers.Scipy()
    optimizer.minimize(
        model.training_loss_closure(data),
        variables=model.trainable_variables,
        method="l-bfgs-b",
        options={"disp": False, "maxiter": MAXITER},
    )


# create multi-output kernel
kernels = [ (PartlySharedIndependentMultiOutput([SquaredExponential()  for _ in range(n_kernels)], nu=nu), 'Custom multi output v2')
             #(SeparateIndependent([SquaredExponential() + Linear() for _ in range(P)]),'Seperate Independent')
            #,(SharedIndependent(SquaredExponential()+Linear(), output_dim=P), 'Shared Independent')
           ]
times = []
for (kernel, name) in kernels:
    start = time.time()
    m = gpflow.models.SVGP_deprecated(kernel, gpflow.likelihoods.Gaussian(), inducing_variable=iv, num_latent_gps=P)
    #print_summary(m)
    optimize_model_with_scipy(m)
    end = time.time()
    times.append((name, end-start))
    #print_summary(m)
    plot_model(m, name)
    print((name, end-start))

print(times)

#--------------------------------------------


def inspect_conditional(inducing_variable_type, kernel_type):
    """
    Helper function returning the exact implementation called
    by the multiple dispatch `conditional` given the type of
    kernel and inducing variable.

    :param inducing_variable_type:
        Type of the inducing variable
    :param kernel_type:
        Type of the kernel

    :return: String
        Contains the name, the file and the linenumber of the
        implementation.
    """
    import inspect
    from gpflow.conditionals import conditional

    implementation = conditional.dispatch(object, inducing_variable_type, kernel_type, object)
    info = dict(inspect.getmembers(implementation))
    return info["__code__"]

Conditional_Shared_independent_iv_shared_independent_kernel = inspect_conditional(SharedIndependentInducingVariables, SharedIndependent)
Conditional_Shared_independent_iv_separate_independent_kernel = inspect_conditional(SharedIndependentInducingVariables, SeparateIndependent)
Conditional_Shared_independent_iv_custom_separate_independent_kernel = inspect_conditional(SharedIndependentInducingVariables, PartlySharedIndependentMultiOutput)
print(Conditional_Shared_independent_iv_custom_separate_independent_kernel)