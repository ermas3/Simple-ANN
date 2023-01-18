import simple_ann as ann
import numpy as np
import matplotlib.pyplot as plt
import time


def test_loss_functions():
    # test MSE
    loss_object = ann._Loss('MSE')
    test_predictions = np.reshape([-2, -1, 0], (1, 3, 1))
    test_labels = np.reshape([-4, 0, 2], (1, 3, 1))
    assert loss_object(test_predictions, test_labels) == 3
    assert np.all(loss_object._prime(test_predictions, test_labels) == np.reshape([4/3, -2/3, -4/3], (1, 3, 1)))

    # test binary_crossentropy
    loss_object = ann._Loss('binary_crossentropy')
    test_predictions = np.reshape([0.2, 0.3, 0.8], (1, 3, 1))
    test_labels = np.reshape([0, 0, 1], (1, 3, 1))
    assert loss_object(test_predictions, test_labels) == -1/3*(np.log(0.8)+np.log(0.7)+np.log(0.8))
    assert np.all(loss_object._prime(test_predictions, test_labels) == np.reshape((test_predictions-test_labels)/(test_predictions*(1-test_predictions)), (1, 3, 1)))

    # test invalid loss function
    try:
        loss_object = ann._Loss('invalid')
        raise Exception('Giving an invalid loss function should have raised a ValueError.')
    except ValueError:
        pass


def test_activation_functions():
    test_input = np.reshape([-2, -1, 0, 5, 1], (1, 5, 1))

    # test relu
    layer_object = ann._Layer(1, 1, activation='relu')
    assert np.all(layer_object._activation(test_input) == np.reshape([0, 0, 0, 5, 1], (1, 5, 1)))
    assert np.all(layer_object._activation_prime(test_input) == np.reshape([0, 0, 0, 1, 1], (1, 5, 1)))

    # test sigmoid
    layer_object = ann._Layer(1, 1, activation='sigmoid')
    assert np.all(layer_object._activation(test_input) == 1 / (1 + np.exp(-test_input)))
    assert np.all(layer_object._activation_prime(test_input) == (1 / (1 + np.exp(-test_input))) * (1 - 1 / (1 + np.exp(-test_input))))

    # test tanh
    layer_object = ann._Layer(1, 1, activation='tanh')
    assert np.all(layer_object._activation(test_input) == np.tanh(test_input))
    assert np.all(layer_object._activation_prime(test_input) == 1 - np.tanh(test_input) ** 2)

    # test linear
    layer_object = ann._Layer(1, 1, activation='linear')
    assert np.all(layer_object._activation(test_input) == test_input)
    assert np.all(layer_object._activation_prime(test_input) == np.reshape([1, 1, 1, 1, 1], (1, 5, 1)))

    # test invalid loss function
    try:
        layer_object = ann._Layer(1, 1, activation='invalid')
        raise Exception('Giving an invalid activation should have raised a ValueError.')
    except ValueError:
        pass


def test_train():
    n = ann.Network(input_shape=2)
    n.add_layer(5, 'relu')
    n.add_layer(5, 'sigmoid')
    n.add_layer(1, 'linear')

    #test invalid input shape
    test_inputs = np.reshape([[-2], [-1], [0], [1], [2], [3]], (6, 1, 1))
    test_labels = np.reshape([[-3], [-1], [1], [3], [5], [7]], (6, 1, 1))
    try:
        n.train(test_inputs, test_labels, 'MSE')
        raise Exception('Giving input of incorrect shape should have raised a ValueError.')
    except ValueError:
        pass

    #test invalid loss function
    test_inputs = np.reshape([[0, 0], [1, 0], [0, 1], [1, 1]], (4, 2, 1))
    test_labels = np.reshape([[0], [1], [1], [0]], (4, 1, 1))
    try:
        n.train(test_inputs, test_labels, 'invalid')
        raise Exception('Giving invalid loss functions should have raised a ValueError.')
    except ValueError:
        pass

    #general testing
    epochs = 2
    assert type(n._train_on_instance(test_inputs[0], test_labels[0], 'MSE', 0.01)) == np.float64
    assert(len(n.train(test_inputs, test_labels, 'MSE', epochs=2))) == epochs
    assert type(n.train(test_inputs, test_labels, 'MSE', epochs=2)) == list


def test_predict():
    n = ann.Network(input_shape=3)
    n.add_layer(5)

    # test incorrect input size
    test_input = np.reshape([1, 2, 3, 4], (1, 4, 1))
    try:
        n.predict(test_input)
        raise Exception('Giving input of incorrect size should have raised a ValueError.')
    except ValueError:
        pass

    # test incorrect input type
    test_input = np.array(['a', 'b', 'c'])
    try:
        n.predict(test_input)
        raise Exception('Giving input of incorrect type should have raised a numpy.core._exceptions.UFuncTypeError.')
    except np.core._exceptions.UFuncTypeError:
        pass

    # test correct output shape
    test_input = np.reshape([1, 2, 3], (1, 3, 1))
    assert np.shape(n.predict(test_input)) == (1, 5, 1)


def test_add_layer():
    n = ann.Network(input_shape=3)
    assert len(n._layers) == 0
    assert n._input_shape == 3
    assert n._trainable_parameters == 0

    # test invalid loss
    try:
        n.add_layer(10, 'invalid')
        raise Exception('Giving an invalid activation should have raised a ValueError.')
    except ValueError:
        pass

    # general testing
    n.add_layer(10, 'relu')
    assert len(n._layers) == 1
    assert n._trainable_parameters == 40


def test_pop_layer():
    # test empty network
    n = ann.Network(input_shape=5)
    try:
        n.pop_layer()
        raise Exception('Popping from an empty network function should have raised an IndexError.')
    except IndexError:
        pass

    # test multi-layer network
    n.add_layer(10, activation='relu')
    n.add_layer(5, activation='sigmoid')
    n.add_layer(6, activation='linear')
    layers_before_pop = n._layers
    assert len(n._layers) == 3
    assert n._input_shape == 5
    assert n._trainable_parameters == 151
    n.pop_layer()
    layers_after_pop = n._layers
    assert layers_after_pop[0] == layers_after_pop[0]
    assert layers_after_pop[1] == layers_after_pop[1]
    assert len(n._layers) == 2
    assert n._trainable_parameters == 115


def test_summary():
    # test empty network
    n = ann.Network(input_shape=5)
    assert n.summary() == 'Total amount of layers: 0\n' \
                          'Total amount of trainable parameters: 0'

    # test single layer network
    n = ann.Network(1)
    n.add_layer(2, activation='relu')
    assert n.summary() == 'Layer 1: Input dimension: 1, Output dimension: 2, Trainable parameters: 4, Activation: relu\n' \
                          'Total amount of layers: 1\n' \
                          'Total amount of trainable parameters: 4'

    # test multi-layer network
    n.add_layer(4, activation='linear')
    assert n.summary() == 'Layer 1: Input dimension: 1, Output dimension: 2, Trainable parameters: 4, Activation: relu\n' \
                          'Layer 2: Input dimension: 2, Output dimension: 4, Trainable parameters: 12, Activation: linear\n' \
                          'Total amount of layers: 2\n' \
                          'Total amount of trainable parameters: 16'

    # test that summary still works after pop
    n.pop_layer()
    assert n.summary() == 'Layer 1: Input dimension: 1, Output dimension: 2, Trainable parameters: 4, Activation: relu\n' \
                          'Total amount of layers: 1\n' \
                          'Total amount of trainable parameters: 4'
