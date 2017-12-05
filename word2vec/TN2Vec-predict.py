# -*- coding: utf-8 -*-
from six.moves import range
import gensim.models.word2vec as w2v
import numpy as np

def nearest_similarity_cosmul(start1, end1, end2):
    similarities = model.most_similar_cosmul(
        positive=[end2, start1],
        negative=[end1]
    )
    start2 = similarities[0][0]
    print("{start1} is related to {end1}, as {start2} is related to {end2}".format(**locals()))
    return start2

model = w2v.Word2Vec.load("tn2vec.w2v")

sim = model.most_similar(u"river".encode("utf-8"))
for idx in range(len(sim)):
    print str(sim[idx][0]) + "\t\t distancia: " + str(sim[idx][1])
print ""

print model.similarity("river","river")

nearest_similarity_cosmul("boca", "river", "millonario")

input = model.wv['river']
sim = np.dot(model.wv['river'], -model.wv.syn0norm.T).argsort()[0:8]
for idx in range(8):
    print model.wv.index2word[sim[idx]]