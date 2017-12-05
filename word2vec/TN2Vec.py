# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

# for word econding
import codecs
# for regex
import glob
import logging
#for conccurency
import multiprocessing
import os
import pprint
# more regex
import re

# natural language toolkit
import nltk
import gensim.models.word2vec as w2v
#dimensionality reduction
import sklearn.manifold
import numpy as np
import pandas as pd
import seaborn as sns


nltk.download("punkt")
nltk.download("stopwords")

book_filenames = sorted(glob.glob("sports-6k/*"))

print("Found books:")
book_filenames

corpus_raw = u""
for book_filename in book_filenames:
    print("Reading '{0}'...".format(book_filename))
    with codecs.open(book_filename, "r", "utf-8") as book_file:
        corpus_raw += book_file.read()
    print("Corpus is now {0} characters long".format(len(corpus_raw)))
    print()

tokenizer = nltk.data.load('tokenizers/punkt/spanish.pickle')
### create sentences
raw_sentences = tokenizer.tokenize(corpus_raw)

#convert into a list of words
#rtemove unnnecessary,, split into words, no hyphens
#list of words
def sentence_to_wordlist(raw):
    regex = u"[^a-zA-Z-Záéíóúñ]"
    clean = re.sub(regex," ", raw).encode("utf-8").lower()
    words = clean.split()
    return words

#sentence where each word is tokenized
sentences = []
for raw_sentence in raw_sentences:
    if len(raw_sentence) > 0:
        sentences.append(sentence_to_wordlist(raw_sentence))

### Print raw sentences,  after being curated and corpus size
print(raw_sentences[5])
print(sentence_to_wordlist(raw_sentences[5]))
token_count = sum([len(sentence) for sentence in sentences])
print("The book corpus contains {0:,} tokens".format(token_count))

del raw_sentence

### Train the model

#ONCE we have vectors
#step 3 - build model
#3 main tasks that vectors help with
#DISTANCE, SIMILARITY, RANKING

# Dimensionality of the resulting word vectors.
#more dimensions, more computationally expensive to train
#but also more accurate
#more dimensions = more generalized
num_features = 300
# Minimum word count threshold.
min_word_count = 3

# Number of threads to run in parallel.
#more workers, faster we train
num_workers = multiprocessing.cpu_count()

# Context window length.
context_size = 14

# Downsample setting for frequent words.
#0 - 1e-5 is good for this
downsampling = 1e-3

# Seed for the RNG, to make the results reproducible.
#random number generator
#deterministic, good for debugging
seed = 1

model = w2v.Word2Vec(
    sg=1,
    seed=seed,
    workers=num_workers,
    size=num_features,
    min_count=min_word_count,
    window=context_size,
    sample=downsampling
)

model.build_vocab(sentences)

### Start training, this might take a minute or two...
model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)

### Save the file
model.save("tn2vec.w2v")

