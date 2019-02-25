# Utilities for the NN exercises in ISTA 421, Introduction to ML

import numpy
import math
import visualize
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------

def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))


# -------------------------------------------------------------------------

def initialize(hidden_size, visible_size):
    """
    Sample weights uniformly from the interval [-r, r] as described in lecture 23.
    Return 1d array theta (in format as described in Exercise 2)
    :param hidden_size: number of hidden units
    :param visible_size: number of visible units (of input and output layers of autoencoder)
    :return: theta array
    """

    ### YOUR CODE HERE ###
    #Assuming autoencoder with one layer- Zavier's initiliazation for each neuron
    r = math.sqrt(6.0 / (hidden_size + visible_size + 1.0))
    w1 = numpy.random.uniform(-r, r, (hidden_size, visible_size))
    w2 = numpy.random.uniform(-r, r, (visible_size, hidden_size))
    b1 = numpy.random.uniform(-r, r, hidden_size, )
    #print("b1 = ", b1,"b1.shape = ", b1.shape)
    b2 = numpy.random.uniform(-r, r, visible_size,)
    theta = numpy.concatenate([w1.flatten(), w2.flatten(), b1.flatten(), b2.flatten()])
    #theta = None  # implement
    #print (theta.shape)
    return theta




# -------------------------------------------------------------------------

def autoencoder_cost_and_grad(theta, visible_size, hidden_size, lambda_, data):
    """
    The input theta is a 1-dimensional array because scipy.optimize.minimize expects
    the parameters being optimized to be a 1d array.
    First convert theta from a 1d array to the (W1, W2, b1, b2)
    matrix/vector format, so that this follows the notation convention of the
    lecture notes and tutorial.
    You must compute the:
        cost : scalar representing the overall cost J(theta)
        grad : array representing the corresponding gradient of each element of theta
    """

    ### YOUR CODE HERE ###
    deltaw1 = 0
    deltaw2 = 0
    deltab1 = 0
    deltab2 = 0
    jtheta = 0

    # Since theta = w1 + w2 + b1 + b2, the order in initialize
    w1 = numpy.reshape(theta[0:visible_size * hidden_size], (hidden_size, visible_size))
    w2 = numpy.reshape(theta[visible_size * hidden_size:2 * visible_size * hidden_size], (visible_size, hidden_size))
    b1 = numpy.reshape(theta[2 * visible_size * hidden_size:2 * visible_size * hidden_size + hidden_size],
                       (hidden_size,))
    b2 = numpy.reshape(theta[2 * visible_size * hidden_size + hidden_size:len(theta)], (visible_size,))

    for i in range(data.shape[1]):
        #calculate the cost J
        # z2 = W1.a1 + b1
        z2 = numpy.dot(w1, data[:, i]) + b1  # activation, a1 is input for first layer
        # a2 = sigmoid(z2)
        a2 = sigmoid(z2)
        # z3 = W2.a2 + b2
        z3 = numpy.dot(w2, a2) + b2
        # a3 = sigmoid(z3)
        a3 = sigmoid(z3)
        jtheta = jtheta + math.pow(numpy.linalg.norm(a3 - data[:, i]), 2)

        #Calculate the gradient delta ( slide 11 leture 25)
        delta3 = -(data[:, i] - a3) * (a3 * (1 - a3))       #to visible layer
        delta2 = numpy.dot(w2.T, delta3) * (a2 * (1 - a2))  #to hidden layer

        # numpy.newaxis adds a column to help transpose
        deltaw1 = deltaw1 + numpy.dot(numpy.array(delta2[:, numpy.newaxis]),
                                      numpy.array(data[:, i][:, numpy.newaxis]).T)
        deltaw2 = deltaw2 + numpy.dot(numpy.array(delta3[:, numpy.newaxis]), numpy.array(a2[:, numpy.newaxis]).T)
        deltab1 = deltab1 + delta2
        deltab2 = deltab2 + delta3

    jtheta = jtheta / (2 * data.shape[1])
    cost = 0
    for i in range(2 * hidden_size * visible_size):
        cost = cost + math.pow(theta[i], 2)  #  weight decay term
    cost = jtheta + (cost * lambda_) / 2.0

    deltaw1 = (deltaw1 / data.shape[1] + lambda_ * w1).flatten()
    deltaw2 = (deltaw2 / data.shape[1] + lambda_ * w2).flatten()
    deltab1 = (deltab1 / data.shape[1]).flatten()
    deltab2 = (deltab2 / data.shape[1]).flatten()
    grad = numpy.concatenate([deltaw1, deltaw2, deltab1, deltab2])    #concatenate since optimizer expects it

    return cost, grad


# -------------------------------------------------------------------------

def autoencoder_cost_and_grad_sparse(theta, visible_size, hidden_size, lambda_, rho_, beta_, data):
    """
    Version of cost and grad that incorporates the hidden layer sparsity constraint
        rho_ : the target sparsity limit for each hidden node activation
        beta_ : controls the weight of the sparsity penalty term relative
                to other loss components

    The input theta is a 1-dimensional array because scipy.optimize.minimize expects
    the parameters being optimized to be a 1d array.
    First convert theta from a 1d array to the (W1, W2, b1, b2)
    matrix/vector format, so that this follows the notation convention of the
    lecture notes and tutorial.
    You must compute the:
        cost : scalar representing the overall cost J(theta)
        grad : array representing the corresponding gradient of each element of theta
    """

    ### YOUR CODE HERE ###
    deltaw1 = 0
    deltaw2 = 0
    deltab1 = 0
    deltab2 = 0
    jtheta = 0

    # theta = w1 + w2 + b1 + b2, the order in initialize
    w1 = numpy.reshape(theta[0:visible_size * hidden_size], (hidden_size, visible_size))
    w2 = numpy.reshape(theta[visible_size * hidden_size:2 * visible_size * hidden_size], (visible_size, hidden_size))
    b1 = numpy.reshape(theta[2 * visible_size * hidden_size:2 * visible_size * hidden_size + hidden_size],
                       (hidden_size,))
    b2 = numpy.reshape(theta[2 * visible_size * hidden_size + hidden_size:len(theta)], (visible_size,))

    roj = numpy.zeros(shape=(hidden_size,))
    gdelta3 = []
    ga2 = []

    for i in range(data.shape[1]):
        # calculate the cost J
        # z2 = W1.a1 + b1
        z2 = numpy.dot(w1, data[:, i]) + b1
        # a2 = sigmoid(z2)
        a2 = sigmoid(z2)
        roj = roj + a2
        # z3 = W2.a2 + b2
        z3 = numpy.dot(w2, a2) + b2
        # a3 = sigmoid(z3)
        a3 = sigmoid(z3)
        jtheta = jtheta + math.pow(numpy.linalg.norm(a3 - data[:, i]), 2)

        # Calculate the gradient delta ( slide 11 leture 25)
        delta3 = -(data[:, i] - a3) * (a3 * (1 - a3))
        gdelta3.append(delta3)
        ga2.append(a2)
        deltaw2 = deltaw2 + numpy.dot(numpy.array(delta3[:, numpy.newaxis]), numpy.array(a2[:, numpy.newaxis]).T)
        deltab2 = deltab2 + delta3

    roj = roj / data.shape[1]

    for i in range(data.shape[1]):
        # delta2, delta3, a2 = current deltas and a's for this data point
        delta3 = gdelta3[i]
        a2 = ga2[i]
        delta2 = numpy.zeros(hidden_size)
        for j in range(hidden_size):
            delta2[j] = (numpy.dot(numpy.array(w2[:, j][:, numpy.newaxis]).T, gdelta3[i]) + beta_ * (-rho_ / roj[j] + ((1 - rho_) / (1 - roj[j])))) * (a2[j] * (1 - a2[j]))

        deltaw1 = deltaw1 + numpy.dot(numpy.array(delta2[:, numpy.newaxis]), numpy.array(data[:, i][:, numpy.newaxis]).T)
        deltab1 = deltab1 + delta2

    # KL divergence
    kl = 0

    for h_index in range(hidden_size):
        kl = kl + rho_ * numpy.log(rho_ / roj[h_index]) + (1 - rho_) * numpy.log((1 - rho_) / (1 - roj[h_index]))


    jtheta = jtheta / (2 * data.shape[1])
    cost = 0
    for i in range(2 * hidden_size * visible_size):
        cost = cost + math.pow(theta[i], 2)
    cost = jtheta + (cost * lambda_) / 2.0 + kl * beta_

    deltaw1 = (deltaw1 / data.shape[1] + lambda_ * w1).flatten()
    deltaw2 = (deltaw2 / data.shape[1] + lambda_ * w2).flatten()
    deltab1 = (deltab1 / data.shape[1]).flatten()
    deltab2 = (deltab2 / data.shape[1]).flatten()
    grad = numpy.concatenate([deltaw1, deltaw2, deltab1, deltab2])

    return cost, grad


# -------------------------------------------------------------------------

def autoencoder_feedforward(theta, visible_size, hidden_size, data):
    """
    Given a definition of an autoencoder (including the size of the hidden
    and visible layers and the theta parameters) and an input data matrix
    (each column is an image patch, with 1 or more columns), compute
    the feedforward activation for the output visible layer for each
    data column, and return an output activation matrix (same format
    as the data matrix: each column is an output activation "image"
    corresponding to the data input).

    Once you have implemented the autoencoder_cost_and_grad() function,
    simply copy the part of the code that computes the feedforward activations
    up to the output visible layer activations and return that activation.
    You do not need to include any of the computation of cost or gradient.
    It is likely that your implementation of feedforward in your
    autoencoder_cost_and_grad() is set up to handle multiple data inputs,
    in which case your only task is to ensure the output_activations matrix
    is in the same corresponding format as the input data matrix, where
    each output column is the activation corresponding to the input column
    of the same column index.

    :param theta: the parameters of the autoencoder, assumed to be in this format:
                  { W1, W2, b1, b2 }
                  W1 = weights of layer 1 (input to hidden)
                  W2 = weights of layer 2 (hidden to output)
                  b1 = layer 1 bias weights (to hidden layer)
                  b2 = layer 2 bias weights (to output layer)
    :param visible_size: number of nodes in the visible layer(s) (input and output)
    :param hidden_size: number of nodes in the hidden layer
    :param data: input data matrix, where each column is an image patch,
                  with one or more columns
    :return: output_activations: an matrix output, where each column is the
                  vector of activations corresponding to the input data columns
    """

    ### YOUR CODE HERE ###
    output_activations = numpy.empty([visible_size, data.shape[1]])  # 784xinput_size output activation

    # theta = w1 + w2 + b1 + b2, the order in initialize
    w1 = numpy.reshape(theta[0:visible_size * hidden_size], (hidden_size, visible_size))
    w2 = numpy.reshape(theta[visible_size * hidden_size:2 * visible_size * hidden_size], (visible_size, hidden_size))
    b1 = numpy.reshape(theta[2 * visible_size * hidden_size:2 * visible_size * hidden_size + hidden_size],
                       (hidden_size,))
    b2 = numpy.reshape(theta[2 * visible_size * hidden_size + hidden_size:len(theta)], (visible_size,))

    for i in range(0, data.shape[1]):
        z2 = numpy.dot(w1, data[:, i]) + b1  # for 1st visible layer, activation a1 is input
        a2 = sigmoid(z2)
        z3 = numpy.dot(w2, a2) + b2
        a3 = sigmoid(z3)
        output_activations[:, i] = numpy.array(a3[:, numpy.newaxis]).T

    return output_activations


# -------------------------------------------------------------------------

def save_model(theta, visible_size, hidden_size, filepath, **params):
    """
    Save the model to file.  Used by plot_and_save_results.
    :param theta: the parameters of the autoencoder, assumed to be in this format:
                  { W1, W2, b1, b2 }
                  W1 = weights of layer 1 (input to hidden)
                  W2 = weights of layer 2 (hidden to output)
                  b1 = layer 1 bias weights (to hidden layer)
                  b2 = layer 2 bias weights (to output layer)
    :param visible_size: number of nodes in the visible layer(s) (input and output)
    :param hidden_size: number of nodes in the hidden layer
    :param filepath: path with filename
    :param params: optional parameters that will be saved with the model as a dictionary
    :return:
    """
    numpy.savetxt(filepath + '_theta.csv', theta, delimiter=',')
    with open(filepath + '_params.txt', 'a') as fout:
        params['visible_size'] = visible_size
        params['hidden_size'] = hidden_size
        fout.write('{0}'.format(params))


# -------------------------------------------------------------------------

def plot_and_save_results(theta, visible_size, hidden_size, root_filepath=None,
                          train_patches=None, test_patches=None, show_p=False,
                          **params):
    """
    This is a helper function to streamline saving the results of an autoencoder.
    The visible_size and hidden_size provide the information needed to retrieve
    the autoencoder parameters (w1, w2, b1, b2) from theta.

    This function does the following:
    (1) Saves the parameters theta, visible_size and hidden_size as a text file
        called '<root_filepath>_model.txt'
    (2) Extracts the layer 1 (input-to-hidden) weights and plots them as an image
        and saves the image to file '<root_filepath>_weights.png'
    (3) [optional] train_patches are intended to be a set of patches that were
        used during training of the autoencoder.  Typically these will be the first
        100 patches of the MNIST data set.
        If provided, the patches will be given as input to the autoencoder in
        order to generate output 'decoded' activations that are then plotted as
        patches in an image.  The image is saved to '<root_filepath>_train_decode.png'
    (4) [optional] test_patches are intended to be a different set of patches
        that were *not* used during training.  This permits inspecting how the
        autoencoder does decoding images it was not trained on.  The output activation
        image is generated the same way as in step (3).  The image is saved to
        '<root_filepath>_test_decode.png'

    The root_filepath is used as the base filepath name for all files generated
    by this function.  For example, if you wish to save all of the results
    using the root_filepath='results1', and you have specified the train_patches and
    test_patches, then the following files will be generated:
        results1_model.txt
        results1_weights.png
        results1_train_decode.png
        results1_test_decode.png
    If no root_filepath is provided, then the output will default to:
        model.txt
        weights.png
        train_decode.png
        test_decode.png
    Note that if those files already existed, they will be overwritten.

    :param theta: model parameters
    :param visible_size: number of nodes in autoencoder visible layer
    :param hidden_size: number of nodes in autoencoder hidden layer
    :param root_filepath: base filepath name for files generated by this function
    :param train_patches: matrix of patches (typically the first 100 patches of MNIST)
    :param test_patches: matrix of patches (intended to be patches NOT used in training)
    :param show_p: flag specifying whether to show the plots before exiting
    :param params: optional parameters that will be saved with the model as a dictionary
    :return:
    """

    filepath = 'model'
    if root_filepath:
        filepath = root_filepath + '_' + filepath
    save_model(theta, visible_size, hidden_size, filepath, **params)

    # extract the input to hidden layer weights and visualize all the weights
    # corresponding to each hidden node
    w1 = theta[0:hidden_size * visible_size].reshape(hidden_size, visible_size).T
    filepath = 'weights.png'
    if root_filepath:
        filepath = root_filepath + '_' + filepath
    visualize.plot_images(w1, show_p=False, filepath=filepath)

    if train_patches is not None:
        # Given: train_patches and autoencoder parameters,
        # compute the output activations for each input, and plot the resulting decoded
        # output patches in a grid.
        # You can then manually compare them (visually) to the original input train_patches
        filepath = 'train_decode.png'
        if root_filepath:
            filepath = root_filepath + '_' + filepath
        train_decode = autoencoder_feedforward(theta, visible_size, hidden_size, train_patches)
        visualize.plot_images(train_decode, show_p=False, filepath=filepath)

    if test_patches is not None:
        # Same as for train_patches, but assuming test_patches are patches that were not
        # used for training the autoencoder.
        # Again, you can then manually compare the decoded patches to the test_patches
        # given as input.
        test_decode = autoencoder_feedforward(theta, visible_size, hidden_size, test_patches)
        filepath = 'test_decode.png'
        if root_filepath:
            filepath = root_filepath + '_' + filepath
        visualize.plot_images(test_decode, show_p=False, filepath=filepath)

    if show_p:
        plt.show()


# -------------------------------------------------------------------------

def get_pretty_time_string(t, delta=False):
    """
    Really cheesy kludge for producing semi-human-readable string representation of time
    y = Year, m = Month, d = Day, h = Hour, m (2nd) = minute, s = second, mu = microsecond
    :param t: datetime object
    :param delta: flag indicating whether t is a timedelta object
    :return:
    """
    if delta:
        days = t.days
        hours = t.seconds // 3600
        minutes = (t.seconds // 60) % 60
        seconds = t.seconds - (minutes * 60)
        return 'days={days:02d}, h={hour:02d}, m={minute:02d}, s={second:02d}' \
                .format(days=days, hour=hours, minute=minutes, second=seconds)
    else:
        return 'y={year:04d},m={month:02d},d={day:02d},h={hour:02d},m={minute:02d},s={second:02d},mu={micro:06d}' \
                .format(year=t.year, month=t.month, day=t.day,
                        hour=t.hour, minute=t.minute, second=t.second,
                        micro=t.microsecond)


# -------------------------------------------------------------------------
