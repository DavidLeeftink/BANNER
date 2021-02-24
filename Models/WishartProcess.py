import numpy as np
import gpflow
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from Likelihoods.WishartProcessLikelihood import WishartProcessLikelihoodBase


class InvWishartProcess(gpflow.models.SVGP):
    # Upgrade guide SVGP Model:
    # https://gpflow.readthedocs.io/en/master/notebooks/gpflow2_upgrade_guide.html#SVGP-Initialiser
    # Short summary: - X and Y should be passed to elbo()/training_loss()
    #                - Z is no longer accepted -> inducing points
    #                - feature parameter renamed to inducing points as well
    #                - minibatch size is done by elbo(): data shape should be in batches to do this.

    def __init__(self, kernel, likelihood=None, D=1, DoF=None, inducing_variable=None):
        # Todo: not clear how the latent GPs work. Currently works only with 1 degree of freedom.
        DoF = D if DoF is None else DoF
        likelihood = InvWishartProcessLikelihood(D, DoF, R=10) if likelihood is None else likelihood

        super().__init__(kernel=kernel,
                         likelihood=likelihood,
                         num_latent_gps=int(D*DoF),
                         inducing_variable=inducing_variable)
