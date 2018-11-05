from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
import time
from torch.autograd import Variable
from filters.CNN import CNN, trainNet
import numpy as np
import torch
from process import *
from filters.BloomFilter import BloomFilter
import sys
import math
from helpers import *

def run(FPR, tau, alpha=0.6185):
  # get the data
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

  # train a model first
  train_loader = get_train_loader(30)
  f = CNN()
  trainNet(f, train_loader, val_loader, batch_size=30, n_epochs=1, learning_rate=0.01)

  FN = 0
  num_shirts = 0
  num_others = 0
  FP = 0
  FN_list = []

  # compute F_p and F_n
  for x in train_list:
    x_str = makestring(list(x[0].numpy().flatten()))
    y=torch.sigmoid(f(x[0].reshape(1,1,28,28)))
    if x_str in S_train:
    	num_shirts += 1
    	if (y[0][1] < tau):
    		FN += 1
    		FN_list.append(x_str)
    else:
    	num_others += 1
    	if (y[0][1] > tau):
    		FP += 1
  print("")
  print("tau is:", tau)
  F_n = FN/float(num_shirts)
  F_p = FP/float(num_others)
  print("F_n is:", F_n)
  print("F_p is:", F_p)

  b = solve_LBF_size(FPR, F_p, F_n, alpha)
  print("b =", b)
  print("over FPR is:", FPR)
  BF = BloomFilter(len(FN_list), alpha **(b / F_n))
  for x in FN_list:
    BF.add(x)

  print("Bloom Filter size:", BF.size / 8)
  print("")
  fps = 0
  total = 0
  fps_adv  = 0
  total_adv = 0
  for x in test_list:
    x_str = makestring(list(x[0].numpy().flatten()))

    # only want to check for false positives
    if x_str in S_train:
      continue

    total += 1
    fp = False
    adv = False
    if (x[1] == 1):
      total_adv += 1
      adv = True
    y = torch.sigmoid(f(x[0].reshape(1,1,28,28)))
    if y[0][1] > tau:
      fps += 1 # F_p
      fp = True
    else:
      result2 = BF.check(x_str)
      if result2 == True:
        fps += 1
        fp = True
    if fp and adv:
      fps_adv += 1

  avg_fp = fps/total
  avg_fp_adv = fps_adv/total_adv
  
  print("non adversarial (all test images)")
  print("fps:", fps, "total:", total)
  print("avg:", avg_fp)
  print("")

  print("adversarial (all test shirts)")
  print("fps:", fps_adv, "total:", total_adv)
  print("avg:", avg_fp_adv)

if __name__ == "__main__":
  run(float(sys.argv[1]), float(sys.argv[2]))
