import simple_ann as ann
import numpy as np
import matplotlib.pyplot as plt


def xor_example():
    learning_rate = 0.1
    epochs = 100
    inputs = np.reshape([[0, 0], [1, 0], [0, 1], [1, 1]], (4, 2, 1))
    labels = np.reshape([[0], [1], [1], [0]], (4, 1, 1))
    n = ann.Network(input_shape=2)
    n.add_layer(5, activation='relu')
    n.add_layer(5, activation='relu')
    n.add_layer(1)
    losses = n.train(inputs, labels, 'MSE', learning_rate=0.01, epochs=epochs, display_info=False)
    plt.plot(losses)
    plt.show()


def linear_example():
    learning_rate = 0.01
    epochs = 100
    inputs = np.reshape([[-2], [-1], [0], [1], [2], [3]], (6, 1, 1))
    labels = np.reshape([[-3], [-1], [1], [3], [5], [7]], (6, 1, 1))
    n = ann.Network(input_shape=1)
    n.add_layer(50, activation='relu')
    n.add_layer(1)
    losses = n.train(inputs, labels, 'MSE', learning_rate=0.01, epochs=epochs, display_info=False)
    plt.plot(losses)
    plt.show()
