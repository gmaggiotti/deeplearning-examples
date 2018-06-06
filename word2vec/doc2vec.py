# -*- coding: utf-8 -*-
# for word econding
import nltk
import gensim
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from os import listdir


path = "notas-sim/"

nltk.download("punkt")
nltk.download("stopwords")

#now create a list that contains the name of all the text file in your data #folder
docLabels = []
docLabels = [f for f in listdir(path) ]
#create a list data that stores the content of all text files in order of their names in docLabels
data = []
for doc in docLabels:
    data.append(open(path + doc).read())


tokenizer = RegexpTokenizer(r'\w+')
stopword_set = set(stopwords.words('spanish'))
#This function does all cleaning of data using two objects above
def nlp_clean(data):
    new_data = []
    for d in data:
        new_str = d.lower()
        dlist = tokenizer.tokenize(new_str)
        dlist = list(set(dlist).difference(stopword_set))
        new_data.append(dlist)
    return new_data


class LabeledLineSentence(object):
    def __init__(self, doc_list, labels_list):
        self.labels_list = labels_list
        self.doc_list = doc_list

    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
            yield gensim.models.doc2vec.LabeledSentence(doc,
                                                        [self.labels_list[idx]])

data = nlp_clean(data)

#iterator returned over all documents
it = LabeledLineSentence(data, docLabels)

model = gensim.models.Doc2Vec(size=300, min_count=0, alpha=0.025, min_alpha=0.025)
model.build_vocab(it)
#training of model
for epoch in range(20):
    print 'iteration '+str(epoch+1)
    model.train(it,total_examples=300, epochs=20)
    model.alpha -= 0.002
    model.min_alpha = model.alpha
#saving the created model
#model.save(‘doc2vec.model’)
print 'model saved'


#start testing
#printing the vector of document at index 1 in docLabels
docvec = model.docvecs[1]
#print docvec

#to get most similar document with similarity scores using document-index
similar_doc = model.docvecs.most_similar(14)
#print similar_doc    ARTICLE-772970


#to get most similar document with similarity scores using document- name
sims = model.docvecs.most_similar('infobae-nota1.txt')
print sims
