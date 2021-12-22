import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import datetime
import time

from src.models.WishartProcess import WishartProcess
from src.likelihoods.WishartProcessLikelihood import WishartLikelihood
from util.training_util import *
from src.kernels.PartlySharedIndependentMOK import PartlySharedIndependentMultiOutput
import tensorflow as tf
import gpflow
from gpflow.utilities import print_summary
from gpflow.kernels import SquaredExponential
from gpflow.inducing_variables import SharedIndependentInducingVariables, InducingPoints
from gpflow.ci_utils import ci_niter

from matplotlib import cm

np.random.seed(2021)
tf.random.set_seed(2021)

plt.rc('axes', titlesize=24)        # fontsize of the axes title
plt.rc('axes', labelsize=18)        # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)       # fontsize of the tick labels
plt.rc('ytick', labelsize=12)       # fontsize of the tick labels
plt.rc('legend', fontsize=20)       # legend fontsize
plt.rc('figure', titlesize=32)      # fontsize of the figure title

print('NumPy version       ', np.__version__)

curdir = os.getcwd()
while not curdir.endswith("BANNER"):
    os.chdir(os.path.dirname(curdir))
    curdir = os.getcwd()

datafolder = 'data/ESMdata2.csv'
df = pd.read_csv(datafolder, delimiter=',')
df = df.rename(columns={'Unnamed: 0': 'id'})

N = len(df.index)
seconds_in_day = 60*60*24
dates_to_datetime = [datetime.strptime(date, '%d/%m/%y') for date in df['date'].values]
measure_times = [datetime.strptime(time, '%H:%M:%S') for time in df['resptime_s'].values]
precise_times = [datetime(year=dates_to_datetime[i].year,
                                   month=dates_to_datetime[i].month,
                                   day=dates_to_datetime[i].day,
                                   hour=measure_times[i].hour,
                                   minute=measure_times[i].minute) for i in range(N)]

timedelta_from_day0 = [time - precise_times[0] for time in precise_times]
days_from_day0 = np.array([td.days + td.seconds / seconds_in_day for td in timedelta_from_day0])
x = days_from_day0

ad_dosage = df['concentrat']
phases = df['phase']

colors_phases = cm.get_cmap('tab10', 10)
phase_labels = ['Baseline',
                'Double blind before reduction',
                'Double blind during reduction',
                'Post medication reduction',
                'Post experiment']

mood_items = ['mood_relaxed', 'mood_down', 'mood_irritat', 'mood_satisfi',
              'mood_lonely', 'mood_anxious', 'mood_enthus', 'mood_suspic',
              'mood_cheerf', 'mood_guilty', 'mood_doubt', 'mood_strong']

mood_needs_shift = np.array([0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0])

colors_mood = cm.get_cmap('rainbow', len(mood_items))

fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(16, 16))
ax = axes[0]
for i, phase in enumerate(np.arange(1, 6)):
    ix_start = df[df['phase'] == phase].first_valid_index()
    ix_end = df[df['phase'] == phase+1].first_valid_index()
    if ix_end is None:
        ix_end = N-1
    ax.axvspan(xmin=x[ix_start], xmax=x[ix_end],
               alpha=0.2, color=colors_phases(i), label=phase_labels[i])
ax.plot(x, ad_dosage, color='k', lw=2, label='AD dosage')
ax.set_ylabel('AD dosage')
ax.set_xlim([0, x[-1]])
ax.legend()
ax.set_title('Anti-depressant dosage across experimental phases')

ax = axes[1]

for i, mood in enumerate(mood_items):
    y = df[mood] + 3*mood_needs_shift[i]  # same Likert everywhere
    ax.plot(x, y, ls=':', color=colors_mood(i), label=mood)
ax.set_xlabel('Time (days)')
ax.set_ylabel('Mood score')
ax.set_xlim([0, x[-1]])
ax.legend(fontsize=14, ncol=3)
ax.set_title('Self-reported mood scores')
plt.show()

# Regression with AD dosage as predictor

# group data by AD dosage
data_per_dosage = dict()
# subset of columns for scalability/interpretability
mood_data_per_dosage = dict()

for c in ad_dosage.unique():
    X_c = df.loc[df['concentrat'] == c].to_numpy()
    X_c_subset = df.loc[df['concentrat'] == c][mood_items].to_numpy()
    data_per_dosage[c] = X_c
    mood_data_per_dosage[c] = X_c_subset
    print(X_c_subset.shape)


# print(len(mood_data_per_dosage), mood_data_per_dosage)
# for data in mood_data_per_dosage:
#     print(data.shape)
# print(len(data_per_dosage), data_per_dosage)


######################################
#####  Wishart Process inference #####
######################################
# Data parameters
X = drug_dosage
Y = mood_data_per_dosage
data = (X,Y)
N, D = X.shape # ()
n_inducing = N
if n_inducing == N:
    Z_init = tf.identity(X)  # X.copy()
else:
    Z_init = np.array([X for _ in range(D)]).T  # .reshape(M,1) # initial inducing variable locations
Z = tf.identity(Z_init)
iv = SharedIndependentInducingVariables(InducingPoints(Z))

# Training params
max_iter = ci_niter(10000)
learning_rate = 0.01
minibatch_size = N # to do: test if this works correctly with lower batch sizes.

# Model parameters
kernel = SquaredExponential(lengthscales=1.)
nu = D+1
R = 10
additive_noise = True
model_inverse = False
multiple_observations = True

# initialize model and likelihood
likelihood = WishartLikelihood(D, nu, R=R,
                               additive_noise=additive_noise,
                               model_inverse=model_inverse,
                               multiple_observations=multiple_observations)
wishart_process = WishartProcess(kernel, likelihood, D=D, nu=nu, inducing_variable=iv)
if n_inducing == N:
    gpflow.set_trainable(wishart_process.inducing_variable, False)
print('wishart process model: (untrained)')
print_summary(wishart_process)

# train model, obtain output
start = time.time()
run_adam(wishart_process, data, max_iter, learning_rate, minibatch_size, natgrads=False, plot=True)
total_time = time.time() - start
print(f'Training time: {total_time}')
print_summary(wishart_process)
print(f"ELBO: {wishart_process.elbo(data):.3}")

n_posterior_samples = 5000
Sigma = wishart_process.predict_mc(X, n_posterior_samples)
mean_Sigma = tf.reduce_mean(Sigma, axis=0)
var_Sigma = tf.math.reduce_variance(Sigma, axis=0)
# np.save('X', X)
# np.save('Y', Y)
# np.save('mean_Sigma', mean_Sigma)
# np.save('var_Sigma', var_Sigma)
# np.save('gt_Sigma', Sigma_gt)
# np.save('10_posterior_samples_Sigma', Sigma[:10])

def plot_marginal_covariance(time, Sigma_mean, Sigma_var, Sigma_gt=None, samples=None):
    N, _, D = Sigma_mean.shape

    f, axes = plt.subplots(nrows=D, ncols=D, figsize=(12, 12))
    for i in range(D):
        for j in range(D):
            if i <= j:
                if Sigma_gt is not None:
                    axes[i, j].plot(time, Sigma_gt[:, i, j], label='Ground truth', color='C0')
                axes[i, j].plot(time, Sigma_mean[:, i, j], label='VB', zorder=-5, color='red')
                # 2 standard deviations from the mean =\approx 95%
                top = Sigma_mean[:, i, j] + 2.0 * Sigma_var[:, i, j] ** 0.5
                bot = Sigma_mean[:, i, j] - 2.0 * Sigma_var[:, i, j] ** 0.5
                # plot std -> to do
                axes[i, j].fill_between(time[:,i], bot, top, color='red', alpha=0.05, zorder=-10, label='95% HDI')
                if samples is not None:
                    axes[i, j].plot(time, samples[:, i, j], label='function samples', zorder=-5, color='red', alpha=0.3)
                if i == j:
                    axes[i, j].set_title('Marginal variance {:d}'.format(i))
                else:
                    axes[i, j].set_title(r'Marginal covariance $\Sigma_{{{:d}{:d}}}$'.format(i, j))
                axes[i, j].set_xlabel('Time')
            else:
               axes[i, j].axis('off')

    plt.subplots_adjust(top=0.9)
    plt.suptitle('BANNER: Marginal $\Sigma(t)$', fontsize=14)

plot_marginal_covariance(X, mean_Sigma, var_Sigma)
plt.show()
