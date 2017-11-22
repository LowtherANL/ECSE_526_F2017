"""Neural Network class, currently only supports fully connected"""
import numpy as np
import layer as fcl


# currently just works for fully connected
# if a different version of a network is created, this should really be a subclass
class NeuralNetwork(object):
    """General neural network class for back-propagation"""
    def __init__(self, layers):
        """Initializer takes a single layer or an iterable of layers, in the order for the network"""
        try:
            self.size = len(layers)
            self.layers = layers
            self.shape = tuple(map(lambda layer: layer.size, layers))
        except TypeError:
            self.size = 1
            self.layers = [layers]
            self.shape = tuple(layers.size)

    def apply(self, sample):
        """Applies an input vector or matrix to the neural net and returns the result"""
        # could be rewritten with reduce as functional
        # probably not worthwhile
        for layer in self.layers:
            sample = layer.apply(sample)
        return sample

    def backprop(self, sample, expected):
        """applies back-propagation for the network"""
        result = self.apply(sample)
        # TODO write back-propagation for network
        # consider case of using multiple exaples at once for vectoriization

    # TODO allow updating and iterative construction
    # TODO add file saving and restoring
    # TODO string and pretty printing functions
