"""
Simple ANN

An API used for building and training sequential artificial neural networks.

An artifical neural network is a computational model used in machine learning.
By optimizing the parameters in the neural network in a process called 'training', the neural network can
'learn' how to map a certain type of input to a certain type of output.

The neural network is essentially a function with a large amount of adjustable parameters. The exact amount of parameters can
vary a lot and depends on the size of the neural network. The optimization process of the parameters is known as 'training' the
neural network and is implemented using gradient descent and backpropagation. There are various variations of the training algorithm,
this API uses stochastic gradient descent (SGD) in order to train the neural network.
"""

import numpy as np
import matplotlib.pyplot as plt
import time


class _Loss():
    def __init__(self, loss):
        if loss == 'MSE':
            self._loss = lambda predictions, labels: np.mean(np.power(labels - predictions, 2))
            self._loss_prime = lambda predictions, labels: 2 * (predictions - labels) / np.size(predictions)

        elif loss == 'binary_crossentropy':
            self._loss = lambda predictions, labels: -np.mean(labels * np.log(predictions) + (1 - labels) * np.log(1 - predictions))
            self._loss_prime = lambda predictions, labels: (predictions - labels) / (predictions * (1 - predictions))

        else:
            raise ValueError("Invalid choice of loss function.")

    def __call__(self, predictions, labels):
        return self._loss(predictions, labels)

    def _prime(self, predictions, labels):
        return self._loss_prime(predictions, labels)


class _Layer():
    def __init__(self, input_shape, output_shape, activation='linear'):
        self._W = np.random.randn(output_shape, input_shape)
        self._b = np.zeros(shape=(output_shape, 1))
        self._activation_name = activation
        self._trainable_parameters = input_shape * output_shape + output_shape

        self._a_prev = None
        self._z_prev = None
        self._activation_prev = None

        self._input_shape = input_shape
        self._output_shape = output_shape

        if activation == 'relu':
            self._activation = lambda x: np.maximum(x, 0)
            self._activation_prime = lambda x: 1*(x > 0)

        elif activation == 'sigmoid':
            self._activation = lambda x: 1 / (1 + np.exp(-x))
            self._activation_prime = lambda x: self._activation(x) * (1 - self._activation(x))

        elif activation == 'tanh':
            self._activation = lambda x: np.tanh(x)
            self._activation_prime = lambda x: 1 - np.tanh(x) ** 2

        elif activation == 'linear':
            self._activation = lambda x: x
            self._activation_prime = lambda x: 1

        else:
            raise ValueError("Invalid choice of activation function.")

    def _forward(self, a_prev, z_prev, activation_prime_prev):
        # store previous layer data for backpropagation
        self._a_prev = a_prev
        self._z_prev = z_prev
        self._activation_prime_prev = activation_prime_prev
        # forward propagation
        z = np.matmul(self._W, a_prev) + self._b
        a = self._activation(z)
        return a, z, self._activation_prime

    def _backward(self, dz, learning_rate):
        # caluclate gradients
        if self._z_prev is not None:  # check if current layer is first layer
            dz_prev = np.matmul(np.transpose(self._W), dz) * self._activation_prime_prev(self._z_prev)
        dW = np.matmul(dz, np.transpose(self._a_prev))
        db = dz
        # update weights
        self._W -= learning_rate * dW
        self._b -= learning_rate * db
        return dz_prev if self._z_prev is not None else None


class Network():
    """Create a network object in order to create a new empty neural network.
    An empty neural network is the identity operator."""
    def __init__(self, input_shape):
        self._layers = []
        self._input_shape = input_shape
        self._trainable_parameters = 0

    def add_layer(self, output_shape, activation='linear'):
        """
        Add new layer at the last position in the neural network.

        Calling this method with all a dense layer at the last position in the neural network. The input size
        is autmatically set to the output size of the previous layer or set to the given input size of the network
        if this is the first layer.

        Args:
          size: a positive integer indicating the amount of wanted neurons in new layer.
          activation: one of the following strings indicating choice of activation function in new layer:
            * 'linear'
            * 'sigmoid'
            * 'relu'
            * 'tanh'
            * 'sigmoid'
            If no string is passed, linear activation function is chosen as default.

        Returns:
          None

        Raises:
          ValueError: if invalid string is passed as activation
        """
        if output_shape == 0:
            raise ValueError("Invalid choice of output shape. Output shape cannot be 0.")
        if len(self._layers) == 0:
            input_shape = self._input_shape
        else:
            input_shape = self._layers[-1]._output_shape
        self._layers.append(_Layer(input_shape, output_shape, activation))
        self._trainable_parameters += input_shape * output_shape + output_shape

    def pop_layer(self):
        """Removes last layer from non-empty network.

        Args:
          None

        Returns:
          None

        Raises:
          IndexError: if trying to pop from a network with no layers.
        """
        self._trainable_parameters -= self._layers[-1]._trainable_parameters
        self._layers.pop()

    def predict(self, input):
        """Computes neural network prediction from given input.

        Args:
          input: numpy array of shape (1, d, 1), where d is the dimension of the input data.

        Returns:
          prediction: numpy array of shape (1, b, 1), where b is the dimension of the output data.

        Raises:
          ValueError: if input array is of incorrect shape.
          numpy.core._exceptions.UFuncTypeError: if input array contains elements of incorrect type.
        """
        prediction = input
        for layer in self._layers:
            prediction, z, activation_prime = layer._forward(prediction, None, None)
        return prediction

    def _train_on_instance(self, inputs, labels, loss, learning_rate):
        loss_object = _Loss(loss)
        a = inputs
        z = None
        activation_prime = None
        # forward propagation
        for layer in self._layers:
            a, z, activation_prime = layer._forward(a, z, activation_prime)
        # caluclate loss and output gradient
        loss = loss_object(a, labels)
        output_gradient = loss_object._prime(a, labels)
        # back propagation
        for layer in reversed(self._layers):
            output_gradient = layer._backward(output_gradient, learning_rate)
        return loss

    def train(self, inputs, labels, loss, learning_rate=0.01, epochs=5, display_info=True):
        """Train neural network using stochastic gradient descent.

        Args:
          inputs: numpy array of shape (i, d, 1) where i is the amount of training instances and
          d is the dimension of each instance.
          labels: numpy array of shape (i, d, 1) where i is the amount of training instances and
          d is the dimension of the network outputs.
          learning_rate: float indicating the chosen learning rate during training.
          epochs: positive integer indicating chosen amount of training epochs.
          loss: one of the following strings indicating choice of activation function in new layer:
            * 'MSE' - Mean Squared Error
            * 'binary_crossentropy' - Binary Crossentropy Error
          display_info: boolean indicating if informative printouts are to be given during training or not.
          If parameter is not passed, defaults to True.

        Returns:
          losses: list containing the average loss from each training epoch

        Raises:
          ValueError: if invalid string loss function name is given
          ValueError: if shape of input does not match input_shape of network.
        """
        number_of_inputs = np.size(inputs, axis=0)
        losses = []
        start_time = time.time()
        print('Training initiating.')
        for i in range(epochs):
            if display_info == True:
                print(f'Epoch {i + 1}/{epochs}', end=', ')
            epoch_loss = 0
            for j in range(number_of_inputs):
                instance_loss = self._train_on_instance(inputs[j], labels[j], loss, learning_rate)
                epoch_loss += instance_loss
            if display_info == True:
                print(f'Loss: {round(epoch_loss/number_of_inputs, 3)}')
            losses.append(epoch_loss)
        elapsed_time = time.time() - start_time
        print(f'Training complete. Duration: {round(elapsed_time, 3)} s')
        return losses

    def summary(self):
        """Get summary of network.

        Get network summary where information is given about order, input size, output size and amount of trainable
        parameters in each layer.

        Args:
          None

        Returns:
          string containing the network summary
        """
        string_lst = []
        for i, layer in enumerate(self._layers):
            string = f"Layer {i + 1}: Input dimension: {layer._input_shape}, Output dimension: {layer._output_shape}, Trainable parameters: {layer._trainable_parameters}, Activation: {layer._activation_name}"
            string_lst.append(string)
        string_lst.append(f'Total amount of layers: {len(self._layers)}')
        string_lst.append(f'Total amount of trainable parameters: {self._trainable_parameters}')
        return "\n".join(string_lst)
