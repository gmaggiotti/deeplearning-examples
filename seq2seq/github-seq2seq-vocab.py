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

empty = 0; eos = 1; unknown = 2
maxlend=25; maxlenh=25
start_idx = eos+1 # first real word
maxlen = maxlend + maxlenh
sample_size = 2000
vocab_size = 500


if not os.path.exists('github_issues.pkl'):
    #read in data sample 2MM rows (for speed of tutorial)
    traindf, testdf = train_test_split(pd.read_csv('github_issues.csv').sample(n=60000), test_size=.10)

    #print out stats about shape of data
    print ('Train:', str(traindf.shape[0]) , 'rows:' , str( traindf.shape[1] ) )
    print ('Train:', str(testdf.shape[0]) , 'rows:' , str( testdf.shape[1] ) )

    # getting the first 2k from the dataset
    train_dataset = traindf.values[0:sample_size]
    test_dataset = testdf.values[0:sample_size]

    with open('github_issues.pkl','wb') as fp:
        pickle.dump((train_dataset,test_dataset),fp,-1)


with open('github_issues.pkl', 'rb') as fp:
    train_dataset, test_dataset = pickle.load(fp)
print('Train',train_dataset.shape)
print('Test',test_dataset.shape)

link, desc, heads = train_dataset.reshape( (-1,sample_size) )
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

#  Clean, tokenize, also, retain only the top 8,000 words in the vocabulary and set the remaining words
#  to 1 which will become common index for rare words
from collections import Counter
from itertools import chain
def get_vocab(lst):
    vocabcount = Counter(w for txt in lst for w in polish_sentence(txt).split())
    vocab = map(lambda x: x[0], sorted(vocabcount.items(), key=lambda x: -x[1]))
    return vocab[:vocab_size], vocabcount


vocab, vocabcount = get_vocab(heads+desc)
print(vocab[:50])

#  Generate dic and inverse dictionary
def get_idx(vocab):
    word2idx = dict((word, idx+start_idx) for idx,word in enumerate(vocab))
    word2idx['<empty>'] = empty
    word2idx['<eos>'] = eos
    word2idx['<unk>'] = unknown
    idx2word = dict((idx,word) for word,idx in word2idx.iteritems())
    return word2idx, idx2word

#   this gets and index number for each word and the other back entry
#   word2idx['the']=45 => idx2word[45]=['the']
word2idx, idx2word = get_idx(vocab)

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

# build a lookup table of index of outside words to index of inside words
Y = []
decoder_target_data = []
for headline in heads:
        y = []
        for token in polish_sentence( headline ).split():
            try:
                y.append( word2idx[token] )
            except:
                y.append(word2idx['<unk>'])
                print('unknown word')
        # For Decoder Input, you don't need the last word as that is only for prediction
        # when we are training using Teacher Forcing.
        Y.append( lpadd( y[:-1] ) )
        # Decoder Target Data Is Ahead By 1
        decoder_target_data.append(lpadd( y[1:]))


X = []
for d in desc:
    x = []
    for token in polish_sentence( d ).split():
        try:
            x.append( word2idx[token] )
        except:
            x.append(word2idx['<unk>'])
            print('unknown word')
    X.append( lpadd( x ) )

#converting all data to numpy before saving
X = np.array(X)
Y = np.array(Y)
decoder_target_data = np.array(decoder_target_data)

with open('github_issues-bundle.pkl','wb') as fp:
    pickle.dump((X,Y,decoder_target_data, idx2word, word2idx),fp,-1)

print('EOP')






