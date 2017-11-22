"""Layer class for making up a fully connected neural network"""
import numpy as np

#To choose: either use scipy for logistic, or numpy.tanh as activation
class Layer(object):
    """Fully connected neural network using vectorized operations"""
    def __init__(self, size=1, inputs=1):
        """Initializer takes number of nodes and expected size of input"""
        self.size = size
        self.input = inputs
        self.weights = np.ndarray((size,inputs))

    def apply(self, input):
        """Computes final result for a given input"""
        activations = self.weights @ input
        return np.tanh(activations)

    #TODO include mathematics for backpropagation
    #consider reformatting to daisy chain and recurse, rather than iterate in neural net

    #TODO implement file serialization and deserialization

    def __str__(self):
        return str(self.weights)

    def __repr__(self):
        return str(self.weights)