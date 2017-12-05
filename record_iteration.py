import glob
from single_dataset_network import *
from roc_collection import *
import csv


def single_construct_network(dataset, steps):
    train, test = load_data(dataset)
    layer_shapes = [train.shape[0]]
    layer_shapes.append(int(layer_shapes[0]))
    layer_shapes.append(int(layer_shapes[1]))
    L1 = layer.LinearLayer(size=layer_shapes[0], inputs=layer_shapes[0])
    L1.step = 0.000001
    L2 = layer.StepLayer(size=layer_shapes[1], inputs=layer_shapes[0], intervals=steps)
    L2.step = 0.00001
    L1.next = L2
    L3 = layer.LinearLayer(size=layer_shapes[0], inputs=layer_shapes[1], slope=1)
    L3.step = 0.0001
    L2.next = L3
    return L1, train, test

if __name__ == '__main__':
    for steps in range(1, 3):
        sets = len(glob.glob('../pythondata/test*')) + 1
        indices = {}
        with open('../tumor_indices.csv', mode='r') as index_list:
            reader = csv.DictReader(index_list)
            for row in reader:
                for k in row:
                    try:
                        indices[int(k)].append(int(row[k]))
                    except KeyError:
                        indices[int(k)] = [int(row[k])]
        collectors = []
        learning_iterations = 1
        for m in range(0, 15, learning_iterations):
            collectors.append(Collector('../longtest/lin-step-lin-' + str(steps) + '-[' + str(m) + '].csv'))
        for i in range(1, sets):
            print('set: ', i)
            NN, training, testing = single_construct_network(i, steps)
            NN.step = NN.step / steps
            lay = NN
            while lay.next is not None:
                lay = lay.next
                lay.step = lay.step / steps
            t = indices[i]
            for collect in collectors:
                train_network(NN, training, learning_iterations)
                test_network(NN, testing, training, t, collect, i)
        for collect in collectors:
            collect.close()
