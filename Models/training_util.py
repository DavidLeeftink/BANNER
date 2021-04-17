import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import gpflow


def run_adam(model, data, iterations, learning_rate=0.01, minibatch_size=25, natgrads=False, plot=False):
    """
    Utility function running the Adam optimizer.

    :param model: GPflow model
    :param data: tuple (X,Y)
    :param interations (int) number of iterations
    :param learning_rate (float)
    :param minibatch_size (int)
    :param natgrads (bool) Optimize variational parameters with natural gradient if True
    :param plot (bool) Plot loss convergence if true.
    """
    logf = []
    if natgrads:
        variational_params = [(model.q_mu, model.q_sqrt)]
        for param in variational_params:
            gpflow.set_trainable(param, False)
        natgrad_opt = gpflow.optimizers.NaturalGradient(gamma=0.001)

    ## mini batches
    # train_dataset = tf.data.Dataset.from_tensor_slices(data).repeat()#.shuffle(N) # minibatch data
    # train_iter = iter(train_dataset.batch(minibatch_size))

    ## one data batch
    train_iter = tuple(map(tf.convert_to_tensor, data))

    training_loss = model.training_loss_closure(train_iter, compile=True)
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    @tf.function
    def optimization_step():
        optimizer.minimize(training_loss, model.trainable_variables)
        if natgrads:
            natgrad_opt.minimize(training_loss, var_list=variational_params)

    n_steps_per_print = 100
    for step in range(iterations):
        optimization_step()
        if step % n_steps_per_print == 0:
            elbo = -training_loss().numpy()
            logf.append(elbo)
            print(f'Iteration {step}/{iterations}. ELBO: {elbo}')

    if plot:
        plt.figure()
        plt.plot(np.arange(iterations)[::n_steps_per_print], logf)
        plt.xlabel("iteration")
        _ = plt.ylabel("ELBO")
        plt.title('Training convergence')
        plt.show()
    return logf