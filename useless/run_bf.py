from filters.BloomFilter import BloomFilter
import math
import numpy as np
from process import *
# generate different data sets

# uniformly chosen between 0 and 10000
def data0(n):
  S = []
  while(len(S) < n):
    x = np.random.randint(0,10000)
    if not(x in S):
      S.append(x)
  return S

# half uniformly chosen between 0 and 10000, and half uniformly chosen between 1000 and 2000 
def data1(n):
  S = []
  while(len(S) < int(n/2)):
    x = np.random.randint(0,10000)
    if not(x in S):
      S.append(x)
      
  while(len(S) < n):
    x = np.random.randint(1000,2000)
    if not (x in S):
      S.append(x)
  return S

# sum of two normals centered at different points
def data2(n):
  S = np.zeros(n)
  S[:int(n/2)] = np.random.normal(1000, 1000, int(n/2))
  S[int(n/2):] = np.random.normal(5000, 2000, int(n/2))
  return list(S)

BF = BloomFilter(100, 0.01)
S = data2(100)
for x in S:
  BF.add(str(x))

avg_fp = 0
for i in range(10):
  S_fake = data2(1000)
  fps = 0
  total = 0
  for x in S_fake:
    result = BF.check(str(x))
    if x not in S:
      total += 1
    if (x not in S) and result==True :
      fps += 1

  print("trial "+str(i)+": ", fps/total)
  avg_fp += fps/total
avg_fp /= 10
print("avg: ", avg_fp)

train_list, test_list, S_train, S_test, S_test_adv = process_shirt_BF()

BF = BloomFilter(len(S_train), 0.01)
for x in S_train:
  BF.add(x)

# empirically check false positive rate for train & test set
# both adversarial and nonadversarial achieve ~0.01 of course
fps = 0
total = 0
for x in S_test:
  result = BF.check(x)
  if x not in S_train:
    total += 1
  if (x not in S_train) and result==True :
    fps += 1
avg_fp = fps/total

print("non adversarial (all test images)")
print("fps:", fps, "total:", total)
print("avg:", avg_fp)
print("="*30)

fps = 0
total = 0
for x in S_test_adv:
  result = BF.check(x)
  if x not in S_train:
    total += 1
  if (x not in S_train) and result==True :
    fps += 1
avg_fp = fps/total

print("adversarial (all test shirts)")
print("fps:", fps, "total:", total)
print("avg:", avg_fp)