# Millad Dagdoni
# 01.03.2019

import numpy as np

KERNEL_VERTICAL = np.array([
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
], dtype=float)

VERTICAL_IMAGE = np.array([
    [0, 0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0],
], dtype=float)

HORIZONTAL_IMAGE = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 1, 1, 1, 0, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 1, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=float)

def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return x > 0

def conv_valid(image, kernel):
    """Return the valid convolution output as a flat vector."""
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    output = []

    for row in range(image_height - kernel_height + 1):
        for col in range(image_width - kernel_width + 1):
            patch = image[row:row + kernel_height, col:col + kernel_width]
            output.append(np.sum(patch * kernel))

    return np.array(output, dtype=float)


def predict(image, kernel, weights):
    features = conv_valid(image, kernel)
    raw_output = np.dot(features, weights)
    prediction = relu(raw_output)
    return prediction, raw_output, features


def train(images, targets, kernel, epochs=200, learning_rate=0.001):
    np.random.seed(1)

    feature_count = conv_valid(images[0], kernel).shape[0]
    weights = np.random.random(feature_count)

    for epoch in range(epochs):
        total_error = 0

        for image, target in zip(images, targets):
            prediction, raw_output, features = predict(image, kernel, weights)

            error = prediction - target
            total_error += error ** 2

            gradient = error * relu_derivative(raw_output) * features
            weights -= learning_rate * gradient

        if epoch % 20 == 0:
            print(f"epoch {epoch:03d} error: {total_error:.6f}")

    return weights


def main():
    images = [VERTICAL_IMAGE, HORIZONTAL_IMAGE]
    targets = [1, 0]

    weights = train(
        images=images,
        targets=targets,
        kernel=KERNEL_VERTICAL,
        epochs=200,
        learning_rate=0.001,
    )

    vertical_prediction, _, _ = predict(VERTICAL_IMAGE, KERNEL_VERTICAL, weights)
    horizontal_prediction, _, _ = predict(HORIZONTAL_IMAGE, KERNEL_VERTICAL, weights)

    print()
    print(f"Vertical image prediction:   {vertical_prediction:.4f}")
    print(f"Horizontal image prediction: {horizontal_prediction:.4f}")
    print("Program end")


if __name__ == "__main__":
    main()
