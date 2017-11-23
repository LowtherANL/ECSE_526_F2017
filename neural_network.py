"""Neural Network class, currently only supports fully connected"""
import numpy as np
import layer as fcl
import random


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

    def stochastic_backprop(self, cases, results, batch_size=1):
        """applies one iteration of stochastic back-propagation for the network"""
        # build batch
        samples = np.ndarray((np.size(cases, axis=0), batch_size))
        expected = np.ndarray((np.size(results, axis=0), batch_size))
        used = set()
        for i in range(batch_size):
            j = random.randint(np.size(cases, axis=1))
            while j in used:
                j = random.randint(np.size(cases, axis=1))
            used.add(j)
            samples[:,i] = cases[:,j]
            expected[:,i] = cases[:,j]
        # apply back-propagation
        self.layers[0].back_propagate(samples,expected)
        # TODO add proper iteration and step size changing

    def add_layer(self, layer):
        """Adds a layer to the current network, on the end"""
        # Indexed addition would need size checking, and might be worth adding later
        self.size += 1
        self.layers.append(layer)
        # splat operator into list with added value, reconvert to tuple
        self.shape = tuple([*self.shape, layer.size])

    # TODO add file saving and restoring
    # TODO string and pretty printing functions
