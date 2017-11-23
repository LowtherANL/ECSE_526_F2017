"""Layer class for making up a fully connected neural network"""
import numpy as np


# To choose: either use scipy for logistic, or numpy.tanh as activation
# Currently, assumes the output will be of size one. This could be changed to get a "strength" indication, but will
# need modification of the back propagation to properly behave.
class Layer(object):
    """Fully connected neural network using vectorized operations"""
    def __init__(self, size=1, inputs=1, next_layer=None):
        """Initializer takes number of nodes and expected size of input"""
        # next_layer parameter, or next attribute should point to the next layer of nodes for back-prop
        self.size = size
        self.input = inputs
        self.weights = np.ndarray((size, inputs))
        self.next = next_layer
        # Get correct step size, possibly in neural network
        self.step = .01

    def apply(self, sample):
        """Computes final result for a given input sample"""
        activations = self.weights @ sample
        return np.tanh(activations)

    def back_propagate(self, sample, expected):
        """Run back-propagation in recursive fashion through layers"""
        result = self.apply(sample)
        if self.next is None:
            error = (expected - result).transpose()
        else:
            error = self.next.back_propagate(result, expected)
        # Double check matrix operations for correct updates
        jacobean = 1 - result**2
        factors = jacobean * (self.weights @ error)
        self.weights = self.weights + (self.step * (factors @ sample).transpose())
        return error

    # TODO implement file serialization and deserialization

    def __str__(self):
        return str(self.weights)

    def __repr__(self):
        return str(self.weights)
