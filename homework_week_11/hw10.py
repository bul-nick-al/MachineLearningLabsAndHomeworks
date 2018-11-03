"""
HOMEWORK:
this is an example of NN with a topology "2-5-1" and sigmoid activation on each layer

Part 1: write 2 more activation functions: tanh and relu.
    a)Implement new NN with the same topology "2-5-1" and relu activation on the first layer
    b)Implement new NN with the same topology "2-5-1" and tanh activation on the first layer

Part 2: make a new NN with a topology 2-5-5-1 by adding one more hidden layer
It means that you need to recalculate blocks with initialization parameters, predict and updating parameters

"""

import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def relu(x):
    return np.maximum(0, x)

# the following three methods are the derivatives of of the activation functions


def sigmoid_derivative(x):
    return x * (1 - x)


def tanh_derivative(x):
    return 1 - x ** 2


def relu_derivative(x):
    return np.vectorize(lambda x: 1 if x > 0 else 0)(x)


def generate_data():
    """
    :return: the data set used in the homework and the labels
    """
    np.random.seed(0)
    # download the dataset
    X, labels = sklearn.datasets.make_moons(200, noise=0.20)
    y = labels.reshape(-1, 1)
    return X, labels, y


def nn_2_5_1(X, y, l1_activation_f, l1_activation_f_deriv):
    """
    encapsulates the logic for building a 2-5-1 nn, training it with BP and prediction
    :param X:
    :param y:
    :param l1_activation_f: the activation function which will be used for the first level
    :param l1_activation_f_deriv: the derivative of the activation function which will be used for the first level
    """
    # topology of our Neural Network: 2-5-1
    # 2: input layer
    # 5: layer 1
    # 1: layer 2
    input_size = 2
    layer1 = 5
    layer2 = 1
    # initialize parameters
    W1 = 2 * np.random.random((input_size, layer1)) - 1  # weights matrix of the first layer
    b1 = 2 * np.random.random((1, layer1)) - 1  # bias vector of the first layer
    W2 = 2 * np.random.random((layer1, layer2)) - 1  # weights matrix of the second layer
    b2 = 2 * np.random.random((1, layer2)) - 1  # bias vector of the second layer

    # lr: learning rate for parameters updating
    lr = 0.1
    for j in range(20000):
        # FEED FORWARD PROPAGATION
        z1 = np.dot(X, W1) + b1  # first layer perceptrons
        a1 = l1_activation_f(z1)  # first layer activation
        z2 = np.dot(a1, W2) + b2  # second layer perceptrons
        a2 = sigmoid(z2)  # second layer activation

        # BACKWARD PROPAGATION
        # C = 1/2 * (a2-y)^2      <-- loss function
        # ∂C/∂a2 = a2 - y         <-- simple calculus exercise
        # partial derivative of sigmoid = a2*(1-a2)
        # other partial derivatives are described above
        a2_delta = (a2 - y) * sigmoid_derivative(a2)
        a1_delta = a2_delta.dot(W2.T) * l1_activation_f_deriv(a1)

        # UPDATE PARAMETERS
        W1 = W1 - lr * X.T.dot(a1_delta)
        b1 = b1 - lr * a1_delta.sum(axis=0, keepdims=True)
        W2 = W2 - lr * a1.T.dot(a2_delta)
        b2 = b2 - lr * a2_delta.sum(axis=0, keepdims=True)

    def predict(X):
        """
        feedforward
        input: X
        output: final layer output
        W1, b1: parameters of the first  layer
        W2, b2: parameters of the second layer
        z: perceptron = ∑w*x + b
        a: activation of perceptron = sigmoid(z) or other for the first layer
        """
        z1 = np.dot(X, W1) + b1  # first layer perceptrons
        a1 = l1_activation_f(z1)  # first layer activation
        z2 = np.dot(a1, W2) + b2  # second layer perceptrons
        a2 = sigmoid(z2)  # second layer activation
        return a2

    _, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)

    axes[0].scatter(X[:, 0], X[:, 1], c=labels, cmap=plt.cm.Spectral)
    axes[0].set_xlim((-1.5, 2.5))
    axes[0].set_ylim((-1, 1.5))

    test_x1 = np.linspace(-1.5, 2.5, 20)
    test_x2 = np.linspace(-1, 1.5, 20)
    for x1 in test_x1:
        for x2 in test_x2:
            y = predict([[x1, x2]])
            color = 'blue' if y > 0.5 else 'red'
            axes[1].scatter(x1, x2, c=color)
    plt.title("2-5-1 with {} ".format(l1_activation_f.__name__))
    plt.show()


def nn_2_5_5_1(X, y, l1_activation_f, l1_activation_f_deriv):
    """
    encapsulates the logic for building a 2-5-5-1 nn, training it with BP and prediction
    :param X:
    :param y:
    :param l1_activation_f: the activation function which will be used for the first level
    :param l1_activation_f_deriv: the derivative of the activation function which will be used for the first level
    """
    # topology of our Neural Network: 2-5-5-1
    # 2: input layer
    # 5: layer 1
    # 5: layer 2
    # 1: layer 3
    input_size = 2
    layer1 = 5
    layer2 = 5
    layer3 = 1

    # initialize parameters
    W1 = 2 * np.random.random((input_size, layer1)) - 1  # weights matrix of the first layer
    b1 = 2 * np.random.random((1, layer1)) - 1  # bias vector of the first layer
    W2 = 2 * np.random.random((layer1, layer2)) - 1  # weights matrix of the second layer
    b2 = 2 * np.random.random((1, layer2)) - 1  # bias vector of the second layer
    W3 = 2 * np.random.random((layer2, layer3)) - 1  # weights matrix of the third layer
    b3 = 2 * np.random.random((1, layer3)) - 1  # bias vector of the third layer

    # lr: learning rate for parameters updating
    lr = 0.1
    for j in range(20000):
        # FEED FORWARD PROPAGATION
        z1 = np.dot(X, W1) + b1  # first layer perceptrons
        a1 = l1_activation_f(z1)  # first layer activation
        z2 = np.dot(a1, W2) + b2  # second layer perceptrons
        a2 = sigmoid(z2)  # second layer activation
        z3 = np.dot(a2, W3) + b3  # third layer perceptrons
        a3 = sigmoid(z3)  # third layer activation

        # BACKWARD PROPAGATION
        # C = 1/2 * (a2-y)^2      <-- loss function
        # ∂C/∂a2 = a2 - y         <-- simple calculus exercise
        # partial derivative of sigmoid = a2*(1-a2)
        # other partial derivatives are described above
        a3_delta = (a3 - y) * sigmoid_derivative(a3)
        a2_delta = a3_delta.dot(W3.T) * sigmoid_derivative(a2)
        a1_delta = a2_delta.dot(W2.T) * l1_activation_f_deriv(a1)

        # UPDATE PARAMETERS
        W1 = W1 - lr * X.T.dot(a1_delta)
        b1 = b1 - lr * a1_delta.sum(axis=0, keepdims=True)
        W2 = W2 - lr * a1.T.dot(a2_delta)
        b2 = b2 - lr * a2_delta.sum(axis=0, keepdims=True)
        W3 = W3 - lr * a2.T.dot(a3_delta)
        b3 = b3 - lr * a3_delta.sum(axis=0, keepdims=True)

    def predict(X):
        """
        feedforward
        input: X
        output: final layer output
        W1, b1: parameters of the first  layer
        W2, b2: parameters of the second layer
        W3, b3: parameters of the third layer
        z: perceptron = ∑w*x + b
        a: activation of perceptron = sigmoid(z) or other for the first layer
        """
        z1 = np.dot(X, W1) + b1  # first layer perceptrons
        a1 = l1_activation_f(z1)  # first layer activation
        z2 = np.dot(a1, W2) + b2  # second layer perceptrons
        a2 = sigmoid(z2)  # second layer activation
        z3 = np.dot(a2, W3) + b3  # third layer perceptrons
        a3 = sigmoid(z3)  # third layer activation
        return a3

    _, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)

    axes[0].scatter(X[:, 0], X[:, 1], c=labels, cmap=plt.cm.Spectral)
    axes[0].set_xlim((-1.5, 2.5))
    axes[0].set_ylim((-1, 1.5))

    test_x1 = np.linspace(-1.5, 2.5, 20)
    test_x2 = np.linspace(-1, 1.5, 20)
    for x1 in test_x1:
        for x2 in test_x2:
            y = predict([[x1, x2]])
            color = 'blue' if y > 0.5 else 'red'
            axes[1].scatter(x1, x2, c=color)
    plt.title("2-5-5-1 with {} ".format(l1_activation_f.__name__))
    plt.show()


def reset_random():
    np.random.seed(0)


reset_random()  # random is reset before gen./train. each nn, otherwise the results are dependent on the execution order
X, labels, y = generate_data()

# the initial code refactored:
nn_2_5_1(X, y, sigmoid, sigmoid_derivative)
reset_random()

# Part 1:
nn_2_5_1(X, y, relu, relu_derivative)  # a)
reset_random()

nn_2_5_1(X, y, tanh, tanh_derivative)  # b)
reset_random()

# Part 2: I implemented 2-5-5-1 nn and also combine it with the three activation functions
nn_2_5_5_1(X, y, sigmoid, sigmoid_derivative)
reset_random()

nn_2_5_5_1(X, y, relu, relu_derivative)
reset_random()

nn_2_5_5_1(X, y, tanh, tanh_derivative)
