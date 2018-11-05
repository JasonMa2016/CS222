import torch
import torchvision
import torchvision.transforms as transforms

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
	# print(len(train_list))
	# print(len(test_list))
	# print(len(S_train))
	# print(len(S_test))
	# print(len(S_test_adv))
	return train_list, test_list, S_train, S_test, S_test_adv
