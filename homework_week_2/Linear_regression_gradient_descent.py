B0_INITIAL_VALUE = 0
B1_INITIAL_VALUE = 1

ALPHA = 0.0001


def gradient_descent(x_array, y_array):
    """

    :param x_array:
    :param y_array:
    :return:
    """
    b1, b0 = B0_INITIAL_VALUE, B1_INITIAL_VALUE
    alpha = ALPHA
    previous_step_size = 1
    precision = 0.00001
    iters = 0  # iteration counter

    while previous_step_size > precision:
        copy_b0, copy_b1 = b0, b1
        b0 -= alpha * b0_partial_derivative(x_array, y_array, copy_b0, copy_b1)
        b1 -= alpha * b1_partial_derivative(x_array, y_array, copy_b0, copy_b1)

        previous_step_size = abs(b0 - copy_b0) + abs(b1 - copy_b1)

    return b1, b0




def b0_partial_derivative(x_array, y_array, b0, b1):
    result = 0

    for x, y in zip(x_array, y_array):
        result += 2 * (y - (b1 * x + b0))

    return - (1 / len(x_array)) * result


def b1_partial_derivative(x_array, y_array, b0, b1):
    result = 0

    for x, y in zip(x_array, y_array):
        result += 2 * x * (y - (b1 * x + b0))

    return - (1 / len(x_array)) * result
