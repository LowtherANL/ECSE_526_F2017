import layer
import neural_network as nn
import numpy as np

# Test a single neuron on its own
L1 = layer.Layer(size=2, inputs=2)
L1.weights = np.asarray([[0,0],[0,0]])
L1.biases = np.asarray([[0],[0]])
print('weights')
print(L1.weights)
test_data = [np.asarray([[1],[0]]), np.asarray([[0],[1]])] # np.random.uniform(-0.5,0.5,(10,1)) # np.ndarray((10,1))
print('Data')
print(test_data[0])
# [[1],[0],[0],[0],[0],[0],[0],[0],[0],[0]]
# test_data.fill(1)
expected = [np.asarray([[.8],[0]]), np.asarray([[0],[0.8]])]
print('Result: ', L1.apply(test_data))
for i in range(10):
    print('iteration: ',i)
    print('result: \n', L1.apply(test_data[i % 2]))
    L1.back_propagate(test_data[i % 2], expected[i % 2])
    print('weights:\n', L1.weights)