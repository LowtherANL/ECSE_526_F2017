import layer
import neural_network as nn
import numpy as np

# Test a single neuron on its own
L1 = layer.Layer(size=1, inputs=10)
print(L1.weights)
test_data =