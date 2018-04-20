from keras.models import Model
from keras.layers import Input, LSTM, GRU, Dense, Embedding, Bidirectional, BatchNormalization
from keras import optimizers
import numpy as np
import os
import cPickle as pickle
from keras.models import load_model
import pandas as pd


with open('github_issues-bundle.pkl', 'rb') as fp:
    X, Y, decoder_target_data, idx2word, word2idx = pickle.load(fp)

with open('github_issues.pkl', 'rb') as fp:
    train_dataset, test_dataset = pickle.load(fp)

print('Train',train_dataset.shape)
print('Test',test_dataset.shape)

seq2seq_Model = load_model('seq2seq_model_keras.h5')


from seq2seq_utils import Seq2Seq_Inference
seq2seq_inf = Seq2Seq_Inference(encoder_preprocessor=X,
                                decoder_preprocessor=Y,
                                seq2seq_model=seq2seq_Model,
                                idx2word= idx2word,
                                word2idx= word2idx)

# this method displays the predictions on random rows of the holdout set
seq2seq_inf.demo_model_predictions(n=1, issue_df=X, threshold=1)


print('EOP')
