"""Neural Network class, currently only supports fully connected"""
import numpy as np
import layer

#currently just works for fully connected
#if a different versin is created, this should really be a subclass
class NeuralNetwork(object):
    """General neural network class for back-propagation"""
    def __init__(self, layers):
        """Initializer takes a single layer or an iterable of layers, in the order for the network"""
        try:
            self.size = len(layers)
            self.layers = layers
            self.shape = tuple(map(lambda L: L.size, layers))
        except TypeError:
            self.size = 1
            self.layers = [layers]
            self.shape = (layers.size)

    def apply(self, input):
        """Applies an input vector or matrix to the neural net and returns the result"""
        #could be rewritten with reduce as functional
        #probably not worthwhile
        for layer in self.layers:
            input = layer.apply(input)
        return input

    def backprop(self, input, expected):
        """applies back-propagation for the network"""
        result = self.apply(input)
        #TODO write back-propagation for network
        #consider case of using multiple exaples at once for vectoriization

    #TODO allow updating and iterative construction
    #TODO add file saving and restoring
    #TODO string and pretty printing functions