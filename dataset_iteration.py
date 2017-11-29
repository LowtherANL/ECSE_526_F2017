import glob
from single_dataset_network import *
from roc_collection import *

if __name__ == '__main__':
    sets = len(glob.glob('../pythondata/test*')) + 1
    collect = Collector('first_test.csv')
    for i in range(1, sets):
        print('set: ', i)
        NN, training, testing = construct_network(i, middle='step')
        t = [9, 11, 13, 15] if i in [3, 4, 6] else [8, 10, 12, 14]
        train_network(NN, training, 10)
        test_network(NN, testing, training, t, collect, i)
        # train_network(NN, training, 10, testing, t)
        # test_network(NN, testing, training, t)
    collect.close()