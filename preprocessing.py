import scipy.io
import os
import numpy as np
import pickle
import glob

np.set_printoptions(threshold=np.nan)
dataDir = "../Data/" #put your directory here

def load_mat_to_numpy(files):
    """
    filename : str, file .mat with the data
    return : if test or train file, the number of the dataset, the matrix (nb of antenna pairs*20)*nb of samples
    """
    train_mat = scipy.io.loadmat(files[0])['train_feature_group_cell']
    test_mat = scipy.io.loadmat(files[1])['test_feature_group_cell']

    num_end = files[0].find('.mat')
    num_start = files[0].rfind('_')
    number_dataset = int(files[0][num_start + 1:num_end])

    train_data = np.stack(*train_mat, axis=1)
    test_data = np.stack(*test_mat, axis=1)
    train_size = train_data.shape
    train_data = train_data.reshape((train_size[0], train_size[1] * train_size[2]), order='C')
    test_data = test_data.reshape((test_data.shape[0], train_size[1] * train_size[2]), order='C')

    total_data = np.concatenate((train_data, test_data), axis=0)

    feature_means = np.mean(total_data, axis=0)
    total_data = total_data - feature_means
    feature_max = np.max(np.abs(total_data), axis=0)
    total_data = .9 * total_data / feature_max

    train_data = total_data[:train_size[0]]
    test_data = total_data[-test_data.shape[0]:]

    return number_dataset, train_data, test_data

if __name__ == '__main__':
    test_files = sorted(glob.glob(dataDir + 'features_test_*'))
    train_files = sorted(glob.glob(dataDir + 'features_train_*'))
    filenames = zip(train_files, test_files)
    for files in filenames :
        # name = str(file)

        nb, train, test = load_mat_to_numpy(files)
        number = int(nb)
        test_filename = '../PythonData/test' + str(number) + '.pck'
        with open(test_filename, mode='wb') as out:
            pickle.dump(test, out)
        train_filename = '../PythonData/train' + str(number) + '.pck'
        with open(train_filename, mode='wb') as out:
            pickle.dump(train, out)
