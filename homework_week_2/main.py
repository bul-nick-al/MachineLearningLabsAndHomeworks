import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from homework_week_2.Linear_regression_gradient_descent import gradient_descent


def generate_data(b1, b0, size, x_range=(-10, 10), noise_mean=0,
                  noise_std=1):
    """
          input:
          b1, b0 - true parameters of data
          size - size of data, numbers of samples
    x_range - tuple of (min, max) x-values
    noise_mean - noise mean value
    noise_std - noise standard deviation

          output:
          data_x, data_y - data features
          """
    noise = np.random.normal(noise_mean, noise_std, size)
    rnd_vals = np.random.rand(size)
    data_x = x_range[1] * rnd_vals + x_range[0] * (1 - rnd_vals)
    data_y = b1 * data_x + b0 + noise

    return data_x, data_y


def animate(data_x, data_y, true_b1, true_b0, b1, b0, x_range=(-10, 10),
            label="Least squares"):
    plt.scatter(data_x, data_y)
    plt.plot([x_range[0], x_range[1]],
             [x_range[0] * true_b1 + true_b0, x_range[1] * true_b1 + true_b0],
             c="r", linewidth=2, label="True")
    plt.plot([x_range[0], x_range[1]],
             [x_range[0] * b1 + b0, x_range[1] * b1 + b0],
             c="g", linewidth=2, label=label)
    plt.legend()
    plt.show()


# Parameters for data generation ###
true_b1 = 2.5
true_b0 = -7
size = 100
x_range = (0, 10)
noise_mean = 0
noise_std = 1


# Generate the data
data_x, data_y = generate_data(true_b1, true_b0, size,
                               x_range=x_range,
                               noise_mean=noise_mean,
                               noise_std=noise_std)


regression_model = LinearRegression()
regression_model.fit([data_x], data_y)
# Predict data's parameters
# b1, b0 = gradient_descent(data_x, data_y)
#
# # Visualize the data
# print("true b1 : {}\ntrue b0 : {}".format(true_b1, true_b0))
# print("calculated b1 : {}\ncalculated b0 : {}".format(b1, b0))
# animate(data_x, data_y, true_b1, true_b0, b1, b0, x_range=x_range)
