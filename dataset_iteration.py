import glob
from single_dataset_network import *
from roc_collection import *
import csv

if __name__ == '__main__':
    sets = len(glob.glob('../pythondata/test*')) + 1
    collect = Collector('random_test.csv')
    indices = {}
    with open('../tumor_indices.csv', mode='r') as index_list:
        reader = csv.DictReader(index_list)
        for row in reader:
            for k in row:
                try:
                    indices[int(k)].append(int(row[k]))
                except KeyError:
                    indices[int(k)] = [int(row[k])]
    for i in range(1, sets):
        print('set: ', i)
        NN, training, testing = construct_network(i, middle='linear')
        t = indices[i]
        #train_network(NN, training, 25)
        test_network(NN, testing, training, t, collect, i)
        # train_network(NN, training, 10, testing, t)
        # test_network(NN, testing, training, t)
    collect.close()