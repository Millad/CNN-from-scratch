# Millad Dagdoni
# 01.03.2019

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1)
# INIT
alpha = 0.04

kernel_vertical = np.array([
    (0, 0, 0, 1, 0, 0, 0),
    (0, 0, 0, 1, 0, 0, 0),
    (0, 0, 0, 1, 0, 0, 0),
    (0, 0, 0, 1, 0, 0, 0),
    (0, 0, 0, 1, 0, 0, 0),
    (0, 0, 0, 1, 0, 0, 0),
    (0, 0, 0, 1, 0, 0, 0,)])

kernel_horiz = np.array([
    (0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0),
    (1, 1, 1, 1, 1, 1, 1),
    (0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0)])

weights = np.random.random(9)

img_data_vertical_line = np.array([
    (0, 0, 0, 1, 1, 1, 0, 0, 0),
    (0, 0, 0, 1, 1, 0, 0, 0, 0),
    (0, 0, 0, 1, 1, 0, 0, 0, 0),
    (0, 0, 0, 1, 1, 1, 0, 0, 0),
    (0, 0, 0, 1, 1, 0, 0, 0, 0),
    (0, 0, 0, 1, 1, 0, 0, 0, 0),
    (0, 0, 0, 1, 1, 1, 0, 0, 0),
    (0, 0, 0, 1, 1, 0, 0, 0, 0),
    (0, 0, 0, 1, 1, 0, 0, 0, 0)])  # (9,9)

img_data_horiz_line = np.array([
    (0, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0, 0),
    (1, 1, 0, 1, 1, 1, 0, 1, 1),
    (1, 1, 1, 1, 1, 1, 1, 1, 1),
    (0, 0, 1, 0, 0, 0, 1, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0, 0)])  # (9,9)

# plt.imshow(img_data, cmap="gray")
# plt.show()

relu = lambda x: (x >= 0) * x  # returns x if x > 0, return 0 otherwise
relu2deriv = lambda x: x > 0  # returns 1 for input > 0, return 0 otherwise


def conv(img_data, kernel):
    layer_0_data_input_list = list()
    for i in range(3):
        for j in range(3):
            layer_0 = img_data[i: (i + 7), j: (j + 7)]
            layer_1 = np.sum(layer_0 * kernel)
            layer_0_data_input_list.append(layer_1)
    return np.array(layer_0_data_input_list)


y = 1
img_dot = []
if __name__ == "__main__":
    for training_run in range(115):
        # Forward propegation
        layer_1 = conv(img_data_vertical_line, kernel_vertical)
        layer_2 = relu(np.dot(layer_1, weights))
        # Cost
        error = (np.sum(layer_2 - y) ** 2)
        # Backpropegation
        layer_2_delta = ((layer_2 - y) * relu2deriv(layer_1))
        weights -= alpha * (layer_2_delta)
        if (training_run % 10 == 9):
            print("[vertikal bilde error]: " + str(error))

# Test HORIZONTAL image should have high error
layer_1 = conv(img_data_horiz_line, kernel_horiz)
layer_2 = relu(np.dot(layer_1, weights))
error = (np.sum(layer_2 - y) ** 2)
print("[HORIZONTAL image should have high rror]: " + str(error))

# print(img_dot.shape)
# plt.imshow(img_dot.reshape(3,3), cmap="gray")
# plt.show()

print("profram end")
# Program SLUTT
# Millad Dagdoni
