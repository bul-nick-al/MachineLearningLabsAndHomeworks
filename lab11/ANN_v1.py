import numpy as np
import matplotlib.pyplot as plt

# Data Matrix (Generate one by Yourself)
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])
print(f"the shape of X is: {X.shape}")

# Class Information
y = np.array([[0, 1, 1, 0]]).T
print(f"the shape of y is: {y.shape}")

W1 = 2 * np.random.random((3, 5)) - 1
W2 = 2 * np.random.random((5, 1)) - 1

print(f"the shape of W1 is: {W1.shape}")
print(f"the shape of W2 is: {W2.shape}")
print(f"Weights after initializing:")
print(W1)
print(W2)

# Gradient descent
for j in range(10000):
    # Forward propagation
    l1 = 1 / (1 + np.exp(-(np.dot(X, W1))))  # the first  layer + sigmoid
    l2 = 1 / (1 + np.exp(-(np.dot(l1, W2))))  # the second layer + sigmoid
    # print(l1.shape)
    # print(l2.shape)

    l2_delta = (y - l2) * (l2 * (1 - l2))
    # print(l2_delta.shape)

    l1_delta = l2_delta.dot(W2.T) * (l1 * (1 - l1))
    # print(l1_delta.shape)

    W1 += X.T.dot(l1_delta)
    W2 += l1.T.dot(l2_delta)
print(f"Weights after training:")
print(W1)
print(W2)


def predict(x):
    l1 = 1 / (1 + np.exp(-(np.dot(x, W1))))
    l2 = 1 / (1 + np.exp(-(np.dot(l1, W2))))
    return l2


# visualization of NN: it solves XOR problem
test_x1 = np.linspace(0, 1, 20)
test_x2 = np.linspace(0, 1, 20)
for x1 in test_x1:
    for x2 in test_x2:
        y = predict([[x1, x2, 1]])
        color = 'red' if y > 0.25 else 'blue'
        plt.scatter(x1, x2, c=color)
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()
