import tensorflow as tf
import gpflow
from gpflow.kernels import MultioutputKernel, Combination, Kernel
from gpflow.inducing_variables import FallbackSeparateIndependentInducingVariables

class SeparateIndependent(MultioutputKernel, Combination):
    """
    - Partly shared: we use a different kernel for each input dimension,
            but all latent GP's that correspond to this input dimension share the same hyperparameter.
            I.e. for D input and DxP output dims, use D independent kernels.
    - Independent: Latents are uncorrelated a priori.
    """

    def __init__(self, kernels, name=None):
        print('hello world')
        super().__init__(kernels=kernels, name=name)

    @property
    def num_latent_gps(self):
        return len(self.kernels)

    @property
    def latent_kernels(self):
        """The underlying kernels in the multioutput kernel"""
        return tuple(self.kernels)

    def K(self, X, X2=None, full_output_cov=True):
        if full_output_cov:
            # Kxxs = []
            # for k in self.kernels:
            #     Kxxs.append(k.K(X,X2))
            # print(Kxxs.shape)
            # Kxxs = tf.stack(Kxxs, axis=2)
            # print(Kxxs.shape)
            # assert 1==2
            Kxxs = tf.stack([k.K(X, X2) for k in self.kernels], axis=2)  # [N, N2, P]
            return tf.transpose(tf.linalg.diag(Kxxs), [0, 2, 1, 3])  # [N, P, N2, P]
        else:
            return tf.stack([k.K(X, X2) for k in self.kernels], axis=0)  # [P, N, N2]


    def K_diag(self, X, full_output_cov=False):
        stacked = tf.stack([k.K_diag(X) for k in self.kernels], axis=1)  # [N, P]
        return tf.linalg.diag(stacked) if full_output_cov else stacked  # [N, P, P]  or  [N, P]

