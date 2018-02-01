import numpy as np
import pandas as pd
import logging
import os
import cPickle as pickle
import re

from sklearn.model_selection import train_test_split
pd.set_option('display.max_colwidth', 500)
logger = logging.getLogger()
logger.setLevel(logging.WARNING)

empty = 0; eos = 1; maxlend=25; maxlenh=25
start_idx = eos+1 # first real word
maxlen = maxlend + maxlenh


if not os.path.exists('github_issues.pkl'):
    #read in data sample 2MM rows (for speed of tutorial)
    traindf, testdf = train_test_split(pd.read_csv('github_issues.csv').sample(n=50000), test_size=.10)

    #print out stats about shape of data
    print ('Train:', str(traindf.shape[0]) , 'rows:' , str( traindf.shape[1] ) )
    print ('Train:', str(testdf.shape[0]) , 'rows:' , str( testdf.shape[1] ) )

    # getting the first 2k from the dataset
    train_dataset = traindf.values[0:2000]
    test_dataset = testdf.values[0:2000]

    with open('github_issues.pkl','wb') as fp:
        pickle.dump((train_dataset,test_dataset),fp,-1)


with open('github_issues.pkl', 'rb') as fp:
    train_dataset, test_dataset = pickle.load(fp)
print('Train',train_dataset.shape)
print('Test',test_dataset.shape)

link, heads, desc = train_dataset.reshape( (-1,2000) )
print('heads:', heads.shape )
print('desc:', desc.shape )

import HTMLParser
def polish_sentence( sentence ):
    p = HTMLParser.HTMLParser()
    sentence = p.unescape(unicode(sentence, "utf-8"))
    sentence = re.sub(u'\n','', sentence)
    sentence = re.sub(u'<[^>]*>nt','', sentence)
    sentence = re.sub(u'<[^>]*>','', sentence)
    sentence = re.sub(u'\[[a-z\_]*embed:.*\]','', sentence)
    sentence = re.sub(u'\[video:.*\]','', sentence)
    sentence = re.sub(u'[\.\[\]\?\,\(\)\!\"\'\\/\:\-]',' ', sentence)
    sentence = re.sub(u'[ ]+',' ', sentence)
    sentence = re.sub(u'%[0-9][a-zA-Z-0-9]', ' ',sentence)
    return sentence

# # build vocabulary
from collections import Counter
from itertools import chain
def get_vocab(lst):
    vocabcount = Counter(w for txt in lst for w in polish_sentence(txt).split())
    vocab = map(lambda x: x[0], sorted(vocabcount.items(), key=lambda x: -x[1]))
    return vocab, vocabcount


vocab, vocabcount = get_vocab(heads+desc)
print(vocab[:50])

def get_idx(vocab):
    word2idx = dict((word, idx+start_idx) for idx,word in enumerate(vocab))
    word2idx['<empty>'] = empty
    word2idx['<eos>'] = eos
    idx2word = dict((idx,word) for word,idx in word2idx.iteritems())
    return word2idx, idx2word

#   this gets and index number for each word and the other back entry
#   word2idx['the']=45 => idx2word[45]=['the']
word2idx, idx2word = get_idx(vocab)

# build a lookup table of index of outside words to index of inside words
Y = []
for headline in heads:
        y = []
        for token in polish_sentence(headline).split():
            try:
                y.append( word2idx[token] )
            except:
                print('word skipped')
        Y.append(y)


X = []
for d in desc:
    x = []
    for token in polish_sentence(headline).split():
        try:
            x.append( word2idx[token] )
        except:
            print('word skipped')
    X.append(x)
print('EOF')

# # Pre-Process Data For Deep Learning
# from ktext.preprocess import processor
# train_body_raw = traindf.body.tolist()
# train_title_raw = traindf.issue_title.tolist()
# # instantiate data processing object
# body_pp = processor(keep_n=8000, padding_maxlen=70)
# # process data
# train_body_vecs = body_pp.fit_transform(train_body_raw)
#
# # Look at one example of processed issue bodies
# print('\noriginal string:\n', train_body_raw[0], '\n')
# print('after pre-processing:\n', train_body_vecs[0], '\n')



def lpadd(x, maxlend=maxlend, eos=eos):
    """left (pre) pad a description to maxlend and then add eos.
    The eos is the input to predicting the first word in the headline
    """
    assert maxlend >= 0
    if maxlend == 0:
        return [eos]
    n = len(x)
    if n > maxlend:
        x = x[-maxlend:]
        n = maxlend
    return [empty]*(maxlend-n) + x + [eos]