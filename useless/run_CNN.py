from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
import time
from torch.autograd import Variable
from filters.CNN import CNN, trainNet
import numpy as np
import torch
from process import *
from filters.BloomFilter import BloomFilter


train_list, test_list, S_train, S_test, S_test_adv = process_shirt()


#Training
n_training_samples = 20000
train_sampler = SubsetRandomSampler(np.arange(n_training_samples, dtype=np.int64))

#Validation
n_val_samples = 5000
val_sampler = SubsetRandomSampler(np.arange(n_training_samples, n_training_samples + n_val_samples, dtype=np.int64))

#Test
n_test_samples = 5000
test_sampler = SubsetRandomSampler(np.arange(n_test_samples, dtype=np.int64))

test_loader = torch.utils.data.DataLoader(test_list, batch_size=4, sampler=test_sampler, num_workers=2)
val_loader = torch.utils.data.DataLoader(train_list, batch_size=128, sampler=val_sampler, num_workers=2)

#DataLoader takes in a dataset and a sampler for loading (num_workers deals with system level memory) 
def get_train_loader(batch_size):
    train_loader = torch.utils.data.DataLoader(train_list, batch_size=batch_size,
                                           sampler=train_sampler, num_workers=2)
    return(train_loader)

train_loader = get_train_loader(30)
print(train_loader)
CNN = CNN()
trainNet(CNN, train_loader, val_loader, batch_size=30, n_epochs=5, learning_rate=0.001)

tau = 0.5

num_correct = 0
num_total = 0
max_probs = []
for i, data in enumerate(test_loader, 0):
  inputs, labels = data
  inputs, labels = Variable(inputs), Variable(labels)
  output_probs = torch.sigmoid(CNN(inputs)).detach().numpy()
  max_probs = max_probs + list(output_probs.max(axis=1))
  outputs = output_probs.argmax(axis=1)
  
  num_total += len(labels)
  num_correct += len(labels) - (outputs-labels).abs().sum()
  
print("CNN accuracy on test data") #### lol....is this right????
print((num_correct).numpy()/num_total)

BF_size = 0
FN_list = []
for x in train_list:
  x_str = makestring(list(x[0].numpy().flatten()))
  y=torch.sigmoid(CNN(x[0].reshape(1,1,28,28)))
  if(y[0][1] < tau) and (x_str in S_train):
    BF_size += 1
    FN_list.append(x_str)

BF = BloomFilter(len(FN_list), 0.01)
for x in FN_list:
  BF.add(x)

# empirically check false positive rate for train & test set using learned BF
# if filter for shirts (which we trained on), fpr is bad - fix this using sandwich
fps = 0
total = 0
probs = []
for x in test_list:
  x_str = makestring(list(x[0].numpy().flatten()))
  y=torch.sigmoid(CNN(x[0].reshape(1,1,28,28)))
  probs.append(y[0][1].detach().numpy())
  if(y[0][1] > tau):
    result = True
  else:
    result = BF.check(x_str)

  if x_str not in S_train:
    total += 1
  if (x_str not in S_train) and result==True :
    fps += 1
avg_fp = fps/total

print("non adversarial (all test images)")
print("fps:", fps, "total:", total)
print("avg:", avg_fp)
print("="*30)

fps = 0
total = 0
for x in test_list:
  if(x[1] == 1):
    x_str = makestring(list(x[0].numpy().flatten()))
    y=torch.sigmoid(CNN(x[0].reshape(1,1,28,28)))
    if(y[0][1] > tau):
      result = True
    else:
      result = BF.check(x_str)
    if x_str not in S_train:
      total += 1
    if (x_str not in S_train) and result==True :
      fps += 1
avg_fp = fps/total

print("adversarial (all test shirts)")
print("fps:", fps, "total:", total)
print("avg:", avg_fp)