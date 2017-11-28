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
    L1.step = 0.00000001
    layers = [L1]
    for i in range(1,len(layer_shapes)):
        L = layer.Layer(size=layer_shapes[i], inputs=layer_shapes[i - 1])
        L.step = L.step / (10 ** (6-i))
        layers[-1].next = L
        layers.append(L)
    for j in range(50):
        L1.back_propagate(train, train)
    evaluation = L1.apply_chain(test)
    # verification = np.max(evaluation, axis=0) - np.min(evaluation, axis=0)
    result = error(evaluation, test, axis=0)

    # print('check eval: ', verification)
    print('result errors: ', result)
    # print('\n', evaluation.transpose()[:,0:5])
    # for l in layers:
    #     print(np.max(np.abs(l.weights)))
    # TODO improve the statistical analysis
    tumors = result[[8, 10, 12, 14]]
    healthy = result[[1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 15]]

    train_result = error(L1.apply_chain(train), train, axis=0)
    threshold = np.max(train_result)

    min_tumor = np.min(tumors)
    max_healthy = np.max(healthy)
    fp = np.count_nonzero(healthy >= min_tumor)
    fn = np.count_nonzero(tumors <= max_healthy)
    print('max fn, fp:', fn, fp)
    fp = np.count_nonzero(healthy >= threshold)
    fn = np.count_nonzero(tumors <= threshold)
    print('realistic fn, fp', fn, fp)