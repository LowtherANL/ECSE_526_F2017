import layer
import neural_network as nn
import numpy as np
import random

#test a single neuron to activate on and
print('And-gate testing')
L1 = layer.Layer(size=1, inputs=2)
L1.weights = np.asarray([[0,0]])
L1.biases = np.asarray([[0]])
print('weights')
print(L1.weights)
test_data = [np.asarray([[1],[1]]), np.asarray([[0],[1]]), np.asarray([[1],[0]]), np.asarray([[0],[0]])] # np.random.uniform(-0.5,0.5,(10,1)) # np.ndarray((10,1))
# print('Data')
# print(test_data[0])
# [[1],[0],[0],[0],[0],[0],[0],[0],[0],[0]]
# test_data.fill(1)
expected = [np.asarray([[.9]]), np.asarray([[0]]), np.asarray([[0]]), np.asarray([[0]])]
for i in range(1000):
    # print('iteration: ',i)
    # print('result: \n', L1.apply(test_data[i % 2]))
    L1.back_propagate(test_data[i % 4], expected[i % 4])
    # print('weights:\n', L1.weights)
print(L1.weights)
print(L1.biases)

# Test two neurons
print('2-2 pass through')
L1 = layer.Layer(size=2, inputs=2)
L1.weights = np.asarray([[0,0],[0,0]])
L1.biases = np.asarray([[0.0],[0.0]])
print('weights')
print(L1.weights)
test_data = [np.asarray([[1],[0]]), np.asarray([[0],[1]])] # np.random.uniform(-0.5,0.5,(10,1)) # np.ndarray((10,1))
# print('Data')
# print(test_data[0])
# [[1],[0],[0],[0],[0],[0],[0],[0],[0],[0]]
# test_data.fill(1)
expected = [np.asarray([[.8],[0]]), np.asarray([[0],[0.8]])]
# print('Result: ', L1.apply(test_data))
for i in range(100):
    # print('iteration: ',i)
    # print('result: \n', L1.apply(test_data[i % 2]))
    L1.back_propagate(test_data[i % 2], expected[i % 2], debug=False)
    # print('weights:\n', L1.weights)
print(L1.weights)
print(L1.biases)

# Test two layers, xor
print('2 layer xor')
L1 = layer.Layer(size=2, inputs=2)
L1.weights = np.asarray([[0.7,0.1],[0.1,0.7]])
L1.biases = np.asarray([[0.0],[0.0]])
print('L1 weights')
print(L1.weights)
print(L1.biases)
L2 = layer.Layer(size=1, inputs=2)
L2.weights = np.asarray([[0.6,0.4]])
L2.biases = np.asarray([[0]])
print('L2 weights')
print(L2.weights)
print(L2.biases)
L1.next = L2
test_data = [np.asarray([[1],[1]]), np.asarray([[0],[1]]), np.asarray([[1],[0]]), np.asarray([[0],[0]])]
expected = [np.asarray([[0]]), np.asarray([[.9]]), np.asarray([[.9]]), np.asarray([[0]])]
# print('Result: ', L1.apply(test_data))
for i in range(1000):
    # print('iteration: ',i)
    # print('result: \n', L1.apply_chain(test_data[i % 4]))
    # L1.back_propagate(test_data[i % 4], expected[i % 4], debug=False)
    L1.back_propagate(test_data[i % 4], expected[i % 4], debug=False)
    # print('weights:\n', L1.weights)
print('L1 weights')
print(L1.weights)
print(L1.biases)
print('L2 weights')
print(L2.weights)
print(L2.biases)
print('Test cases')
for i in range(4):
    print('case: ', test_data[i].transpose())
    print('output: ', L1.apply_chain(test_data[i]))

# Test two layers, xor batch
print('2 layer xor batch')
L1 = layer.Layer(size=2, inputs=2)
L1.weights = np.asarray([[0.7,0.1],[0.1,0.7]])
L1.biases = np.asarray([[0.0],[0.0]])
print('L1 weights')
print(L1.weights)
print(L1.biases)
L2 = layer.Layer(size=1, inputs=2)
L2.weights = np.asarray([[0.6,0.4]])
L2.biases = np.asarray([[0]])
print('L2 weights')
print(L2.weights)
print(L2.biases)
L1.next = L2
test_data = np.asarray([[1,0,1,0],[1,1,0,0]])
expected = np.asarray([[0,1,1,0]])
# print('Result: ', L1.apply(test_data))
for i in range(1000):
    # print('iteration: ',i)
    # print('result: \n', L1.apply_chain(test_data[i % 4]))
    # L1.back_propagate(test_data[i % 4], expected[i % 4], debug=False)
    L1.back_propagate(test_data, expected, debug=False)
    # print('weights:\n', L1.weights)
print('L1 weights')
print(L1.weights)
print(L1.biases)
print('L2 weights')
print(L2.weights)
print(L2.biases)
print('Test cases')
for i in range(4):
    print('case: ', test_data[i].transpose())
    print('output: ', L1.apply_chain(test_data[i]))