import numpy as np
import layer
import pickle


DEFAULT_SET = 1
DATAPATH = '../PythonData/'
EXT = '.pck'

def load_data(set_number):
    with open(DATAPATH + 'train' + str(set_number) + EXT, mode='rb') as f_train:
        training = pickle.load(f_train)
    with open(DATAPATH + 'test' + str(set_number) + EXT, mode='rb') as f_test:
        testing = pickle.load(f_test)
    return training.transpose(), testing.transpose()


def error(input, output, axis):
    return np.sum(np.power(input - output, 2), axis=axis)


def construct_network(dataset):
    train, test = load_data(dataset)
    layer_shapes = [train.shape[0]]
    layer_shapes.append(int(layer_shapes[0] / 2))
    layer_shapes.append(int(layer_shapes[1] / 5))
    layer_shapes.append(layer_shapes[1])
    layer_shapes.append(layer_shapes[0])
    L1 = layer.Layer(size=layer_shapes[0], inputs=layer_shapes[0])
    L1.step = 0.00000001
    layers = [L1]
    for i in range(1, len(layer_shapes)):
        L = layer.Layer(size=layer_shapes[i], inputs=layer_shapes[i - 1])
        L.step = L.step / (10 ** (6 - i))
        layers[-1].next = L
        layers.append(L)
    return L1, train, test


def train_network(NN, data, iterations):
    for i in range(iterations):
        NN.back_propagate(data, data)
        train_error = error(NN.apply_chain(data), data, axis=0)
        print('max: ', np.max(train_error))
    return NN


def test_network(NN, test, train, tumor):
    evaluation = NN.apply_chain(test)
    result = error(evaluation, test, axis=0)
    print('result errors: ', result)
    # TODO improve the statistical analysis
    # TODO IPORTANT fix the labelling to be corect samples
    tumors = result[tumor]
    h = list(set(range(15)) - set(tumor))
    healthy = result[h]

    train_result = error(NN.apply_chain(train), train, axis=0)
    print('training errors: ', train_result)
    threshold = np.max(train_result)
    print('threshold: ', threshold)
    min_tumor = np.min(tumors)
    max_healthy = np.max(healthy)
    fp = np.count_nonzero(healthy >= min_tumor)
    fn = np.count_nonzero(tumors <= max_healthy)
    print('max fn, fp:', fn, fp)
    fp = np.count_nonzero(healthy >= threshold)
    fn = np.count_nonzero(tumors <= threshold)
    print('realistic fn, fp', fn, fp)


if __name__ == '__main__':
    NN, training, testing = construct_network(DEFAULT_SET)
    train_network(NN, training, 25)
    test_network(NN, testing, training)