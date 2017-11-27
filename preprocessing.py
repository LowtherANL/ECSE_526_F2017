import scipy.io
import os
import numpy as np
import pickle
np.set_printoptions(threshold=np.nan)
dataDir = "dataset/" #put your directory here

def load_mat_to_numpy(filename):
	"""
	filename : str, file .mat with the data
	return : if test or train file, the number of the dataset, the matrix (nb of antenna pairs*20)*nb of samples
	"""
	mat = scipy.io.loadmat(filename)
	if 'train_feature_group_cell' in mat:
		if name[16] == '.':
			number_dataset = name[15]
		else:
			number_dataset = name[15:17]
		train_data = mat['train_feature_group_cell']
		train_data_fin = []
		for i in range(len(train_data[0][0])):
			train_data_reshape = np.zeros((len(train_data[0]), len(train_data[0][0][0])))
			for j in range(len(train_data[0])):
				train_data_reshape[j][:]= train_data[0][j][i]
			train_data_fin.append(train_data_reshape)
		return('train',number_dataset, train_data_fin)
	else:
		if name[15] == '.':
			number_dataset = name[14]
		else:
			number_dataset = name[14:16]
		test_data = mat['test_feature_group_cell']
		print(test_data)

		test_data_fin = []
		for i in range(len(test_data[0][0])):
			test_data_reshape = np.zeros((len(test_data[0]), len(test_data[0][0][0])))
			for j in range(len(test_data[0])):
				test_data_reshape[j][:]= test_data[0][j][i]
			test_data_fin.append(test_data_reshape)
		return('test',number_dataset, test_data_fin)

def normalization(dataset):
	"""
	normalize the dataset feature by feature
	dataset : ndarray of size (nb of antenna pairs*20) 
	return: dataset normalized
	"""
	mean = 0
	#std = 0
	for k in range(len(dataset[0])):
		for i in range(len(dataset[0][0])):
			mean = 0
			std = 0
			maxi = -10
			for j in range(len(dataset)):
				mean += dataset[j][k][i]	
				maxi = max(maxi, abs(dataset[j][k][i])	)
			mean /= len(dataset)
			#for j in range(len(dataset)):
			#	std += (dataset[j][k][i] - mean)**2
			#std /= (len(dataset)-1)
			#std = np.sqrt(std)
			for j in range(len(dataset)):
				dataset[j][k][i] -= mean
				#dataset[j][k][i] /= std
				dataset[j][k][i] /= abs(maxi)
				dataset[j][k][i] = .9*dataset[j][k][i];
	return(dataset)
	

if __name__ == '__main__':
	for file in os.listdir( dataDir ) :
		name = str(file)

		typedata,nb, data = load_mat_to_numpy( dataDir+file )
		number = int(nb)
		data = normalization(data) #if you want to normalize the data
		filename = typedata + '%s' %number +'.py'
		print(filename)
		with open(filename, mode='wb') as out:
			pickle.dump(data, out)

