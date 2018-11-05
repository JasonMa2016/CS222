import numpy as np
import torch
from process import *
from filters.BloomFilter import BloomFilter
import sys
import math
from helpers import *

def run(FPR, alpha=0.6185):
	# get the data
	train_list, test_list, S_train, S_test, S_test_adv = process_shirt()
	BF = BloomFilter(len(S_train), FPR)
	for x in S_train:
		BF.add(x)
	print("Bloom Filter size:", BF.size / 8)
	print("")
	
	fps = 0
	total = 0
	fps_adv = 0
	total_adv = 0
	
	for x in test_list:
		x_str = makestring(list(x[0].numpy().flatten()))
		if x_str in S_train:
			continue
		total += 1
		adv = False
		fp = False
		if (x[1] == 1):
			total_adv += 1
			adv = True
		result = BF.check(x_str)
		if result == True:
			fps += 1
			fp = True
		if adv and fp:
			fps_adv += 1

	avg_fp = fps/total
	avg_fp_adv = fps_adv/total_adv

	print("non adversarial (all test images)")
	print("fps:", fps, "total:", total)
	print("avg:", avg_fp)
	print("="*30)
	print("adversarial (all test shirts)")
	print("fps:", fps_adv, "total:", total_adv)
	print("avg:", avg_fp_adv)

if __name__ == "__main__":
	run(float(sys.argv[1]))