"""Layer class for making up a fully connected neural network"""
import numpy as np


# To choose: either use scipy for logistic, or numpy.tanh as activation
# Currently, assumes the output will be of size one. This could be changed to get a "strength" indication, but will
# need modification of the back propagation to properly behave.
class Layer(object):
    """Fully connected neural network using vectorized operations"""
    # TODO implement correct biasing terms
    def __init__(self, size=1, inputs=1, next_layer=None):
        """Initializer takes number of nodes and expected size of input"""
        # next_layer parameter, or next attribute should point to the next layer of nodes for back-prop
        self.size = size
        self.input = inputs
        # self.weights = np.ndarray((size, inputs))
        weight_range = np.sqrt(3 / inputs)
        self.weights = np.random.uniform(-weight_range, weight_range, (size, inputs))
        self.biases = np.random.uniform(-weight_range, weight_range, (size, 1))
        self.next = next_layer
        # Get correct step size, possibly in neural network
        self.step = .01

    def apply(self, sample):
        """Computes final result for a given input sample"""
        activations = (self.weights @ sample) + self.biases
        return np.tanh(activations)

    def apply_chain(self, sample):
        output = self.apply(sample)
        if self.next is not None:
            return self.next.apply_chain(output)
        else:
            return output

    def back_propagate(self, sample, expected, debug=False):
        """Run back-propagation in recursive fashion through layers"""
        result = self.apply(sample)
        if self.next is None:
            error = (result - expected).transpose()
        else:
            error = self.next.back_propagate(result, expected, debug)
        # Double check matrix operations for correct updates
        if debug:
            print('error: \n', error)
        jacobean = (1 - (result**2)).transpose()
        if debug:
            print('jac: \n', jacobean)
        factors = jacobean * error
        if debug:
            print('factors: \n', factors)
        if debug:
            print('weighted change: \n', sample @ factors)
        # print(self.weights)
        out = error @ self.weights
        self.weights = self.weights - (self.step * (sample @ factors).transpose())
        # print(factors.transpose())
        # print(np.mean(factors, axis=0, keepdims=True))
        self.biases = self.biases - (self.step * np.mean(factors, axis=0, keepdims=True).transpose())
        if debug:
            print(self.weights)
            print(self.biases)
        return out

    # TODO implement file serialization and deserialization

    def __str__(self):
        return str(self.weights)

    def __repr__(self):
        return str(self.weights)


class LinearLayer(Layer):
    """Layer with linear activation function"""
    def __init__(self, size=1, inputs=1, slope=1, next_layer=None):
        super().__init__(size, inputs, next_layer)
        self.slope = 1
        self.jacobean = np.ndarray((1, size))
        self.jacobean.fill(slope)

    def apply(self, sample):
        """Computes final result for a given input sample"""
        activations = (self.weights @ sample) + self.biases
        return np.clip(self.slope * activations, -1, 1)

    def back_propagate(self, sample, expected, debug=False):
        """Run back-propagation in recursive fashion through layers"""
        result = self.apply(sample)
        if self.next is None:
            error = (result - expected).transpose()
        else:
            error = self.next.back_propagate(result, expected, debug)
        factors = self.jacobean * error
        out = error @ self.weights
        self.weights = self.weights - (self.step * (sample @ factors).transpose())
        self.biases = self.biases - (self.step * np.mean(factors, axis=0, keepdims=True).transpose())
        if debug:
            print(self.weights)
            print(self.biases)
        return out


class StepLayer(Layer):
    """Layer with tiered activation function"""
    def __init__(self, size=1, inputs=1, intervals=1, next_layer=None):
        super().__init__(size, inputs, next_layer)
        self.intervals = intervals
        self.step_width = 2 / intervals
        self.step_height = 2 / intervals
        steps = [-1.0]
        for i in range(1, intervals + 1):
            steps.append((i * self.step_height) - 1)
        steps.append(1.0)
        self.steps = np.asarray(steps)
        self.max = intervals + 1
        self.jacobean = np.ndarray((1, size))
        self.jacobean.fill(1)

    def apply(self, sample):
        indices = self.weights * sample
        indices = (indices + 1) / self.intervals
        indices = np.clip(indices, 0, self.max)
        return self.steps[indices]

    def back_propagate(self, sample, expected, debug=False):
        """Run back-propagation in recursive fashion through layers"""
        result = self.apply(sample)
        if self.next is None:
            error = (result - expected).transpose()
        else:
            error = self.next.back_propagate(result, expected, debug)
        factors = self.jacobean * error
        out = error @ self.weights
        self.weights = self.weights - (self.step * (sample @ factors).transpose())
        self.biases = self.biases - (self.step * np.mean(factors, axis=0, keepdims=True).transpose())
        if debug:
            print(self.weights)
            print(self.biases)
        return out