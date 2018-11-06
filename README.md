# CS222
CS222 final project

## Environment Set up:
1. Make sure python3 is correctly installed locally. Version # above 3.6.1, ideally.
2. Install Pytorch and relative dependencies locally:
	- (Mac) conda install pytorch torchvision -c pytorch
	- (Windows) conda install pytorch -c pytorch; pip3 install torchvision

## Brief Overview
/filters contains the basic Bloom Filter and learning model (CNN) implementations.

All tests currently run on the shirts data set.

BF.py tests a basic BF model given a desired FPR rate to be achieved.
example command: python3 BF.py 0.01

LBF.py tests a LBF model given a desired FPR rate to be achieved (data set dependent) and tau. 
example command: python3 LBF.py 0.01 0.5

SBF.py tests a SBF model given a deisred FPR rate to be achieved (data set dependent) and tau.
example command: python3 SBF.py 0.005 0.5

SBF_tau.py tests the same learning model + SBF with different cutoff value tau.
example command: python3 SBF_tau.py 0.005