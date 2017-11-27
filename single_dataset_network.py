import numpy as np
import layer
import pickle


DEFAULT_SET = 1
DATAPATH = '../PythonData/'
EXT = '.pck'

def load_data(set_number):
    with open(DATAPATH + 'train' + str(set_number) + EXT, mode='rb') as f_train:
        training = pickle.load(f_train)
    with open(DATAPATH + 'test' + str(DEFAULT_SET) + EXT, mode='rb') as f_test:
        testing = pickle.load(f_test)
    return training.transpose(), testing.transpose()


def error(input, output, axis):
    return np.sum(np.power(input - output, 2), axis=axis)


if __name__ == '__main__':
    train, test = load_data(DEFAULT_SET)
    layer_shapes = [train.shape[0]]
    layer_shapes.append(int(layer_shapes[0] / 2))
    layer_shapes.append(int(layer_shapes[1] / 5))
    layer_shapes.append(layer_shapes[1])
    layer_shapes.append(layer_shapes[0])
    L1 = layer.Layer(size=layer_shapes[0], inputs=layer_shapes[0])
    layers = [L1]
    for i in range(1,len(layer_shapes)):
        L = layer.Layer(size=layer_shapes[i], inputs=layer_shapes[i - 1])
        layers[-1].next = L
        layers.append(L)
    for j in range(300):
        L1.back_propagate(train, train)
    evaluation = L1.apply_chain(test)
    verification = np.max(evaluation, axis=0) - np.min(evaluation, axis=0)
    result = error(evaluation, test, axis=0)

    print('check eval: ', verification)
    print('result errors: ', result)
    print('\n', evaluation.transpose()[:,0:5])