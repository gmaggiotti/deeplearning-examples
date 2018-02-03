import numpy as np
import os
import cPickle as pickle

with open('github_issues-bundle.pkl', 'rb') as fp:
    X,Y, idx2word, word2idx = pickle.load(fp)

print('EOP')
