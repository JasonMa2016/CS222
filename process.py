import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np 

torch.set_default_tensor_type('torch.FloatTensor')

def makestring(l):
	'''
	args:
	l (list)
	'''
	return (",".join(map(str,l)))

classes = ['shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'boot']

# label and transform shirt data

def process_shirt():
	train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
	test_set = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
	
	K = [] # the set of positive keys in training set
	U = [] # the set of negative keys in training set
	train_list = []
	test_list = []

	S_train = []
	S_test = []
	S_test_adv = []

	for i in range(len(train_set)):
	  if(train_set[i][1] > 0):
	    train_list.append((train_set[i][0], 0))
	  else:
	    train_list.append((train_set[i][0], 1))
	    S_train.append(makestring(train_list[i][0][0].numpy().flatten()))
	    
	for i in range(len(test_set)):
	  if(test_set[i][1] > 0):
	    test_list.append((test_set[i][0], 0))
	  else:
	    test_list.append((test_set[i][0], 1))
	    S_test_adv.append(",".join(map(str,list(test_list[i][0][0].numpy().flatten()))))
	  S_test.append(makestring(test_list[i][0][0].numpy().flatten()))

	return train_list, test_list, S_train, S_test, S_test_adv

def process_url():
	data = torch.tensor(np.loadtxt("data/phishing.csv", delimiter= ",", skiprows=1))
	num_features = data.shape[1]
	y = data[:, num_features-1]
	X = data[:, :num_features-1]

	cutoff = int(X.shape[0] * 0.8)
	X_train, y_train = X[:cutoff], y[:cutoff]
	X_test, y_test = X[cutoff:], y[cutoff:]

	train_list = []
	test_list = []
	S_train = []
	S_test = [] # we don't need this
	S_test_adv = [] # we don't need  this 
	for i in range(len(y_train)):
		if (y_train[i] == 1):
			train_list.append((X_train[i], 1))
			S_train.append(makestring(X_train[i].numpy().flatten()))
		else:
			train_list.append((X_train[i], 0))

	for i in range(len(y_test)):
		test_list.append((X_test[i], y_test[i]))

	return train_list, test_list, S_train, S_test, S_test_adv