import glob
from single_dataset_network import *

if __name__ == '__main__':
    sets = len(glob.glob('../pythondata/test*')) + 1
    for i in range(1, sets):
        print('set: ', i)
        NN, training, testing = construct_network(i)
        train_network(NN, training, 25)
        t = [9, 11, 13, 15] if i in [3, 4, 6] else [8, 10, 12, 14]
        test_network(NN, testing, training, t)