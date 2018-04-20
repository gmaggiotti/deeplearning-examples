import pandas as pd
import logging
import glob
from sklearn.model_selection import train_test_split
pd.set_option('display.max_colwidth', 500)
logger = logging.getLogger()
logger.setLevel(logging.WARNING)


#read in data sample 2M rows (for speed of tutorial)
traindf, testdf = train_test_split(pd.read_csv('github_issues.csv').sample(n=50000),
                                   test_size=.10)


#print out stats about shape of data
print('Train: ' , traindf.shape[1])
print('Test: ' , testdf.shape[1])

# preview data
traindf.head(3)

train_body_raw = traindf.body.tolist()
train_title_raw = traindf.issue_title.tolist()
#preview output of first element
train_body_raw[0]

from ktext.preprocess import processor
# Clean, tokenize, and apply padding / truncating such that each document length = 70
#  also, retain only the top 8,000 words in the vocabulary and set the remaining words
#  to 1 which will become common index for rare words
body_pp = processor(keep_n=8000, padding_maxlen=70)
train_body_vecs = body_pp.fit_transform(train_body_raw)
