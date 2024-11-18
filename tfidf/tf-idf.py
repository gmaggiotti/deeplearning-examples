import pandas as pd
from numpy.ma.extras import unique
from sklearn.feature_extraction.text import TfidfVectorizer
import math

docummentA ="the man went out for a walk"
docummentB ="the children sat around the fire"

bagOfWordsA = docummentA.split(" ")
bagOfWordsB = docummentB.split(" ")

uniqueWords = set(bagOfWordsA).union(set(bagOfWordsB))

numOfWordsA =dict.fromkeys(uniqueWords, 0)
for word in bagOfWordsA:
    numOfWordsA[word] += 1
numOfWordsB =dict.fromkeys(uniqueWords, 0)
for word in bagOfWordsB:
    numOfWordsB[word] +=1

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# removing stop words
# filteredA = [w for w in bagOfWordsA if not w in stop_words]
# filteredB = [w for w in bagOfWordsB if not w in stop_words]
# uniqueWords = set(filteredA).union(set(filteredB))
# numOfWordsA =dict.fromkeys(uniqueWords, 0)
# for word in filteredA:
#     numOfWordsA[word] += 1
# numOfWordsB =dict.fromkeys(uniqueWords, 0)
# for word in filteredB:
#     numOfWordsB[word] +=1

# tf(t,d) = #word / #total_words_in_doc
# number of rep of each word divided by # of total words per doc
def computeTF(wordDict, bagOfWords):
    tfDict = {}
    N = len(bagOfWords)
    for word in wordDict.keys():
        tfDict[word] = wordDict[word]/N
    return tfDict

tfA = computeTF(numOfWordsA, bagOfWordsA)
tfB = computeTF(numOfWordsB, bagOfWordsB)
print("tfA", tfA)
print("tfB", tfB)

def computeIDF(documents: [dict]):
    N = len(documents)
    idfDict = dict.fromkeys(documents[0].keys(), 0)
    for document in documents:
        for word, val in document.items():
            if val > 0:
                idfDict[word] += 1
    for word, val in idfDict.items():
        idfDict[word] = math.log(N/float(val))
    return idfDict

idfs = computeIDF([numOfWordsA, numOfWordsB])
print("idfs",idfs)

def computeTFIDF(tfBagOfWords, idfs):
    tfidf = {}
    for word, val in tfBagOfWords.items():
        tfidf[word] = val * idfs[word]
    return tfidf

tfidfA = computeTFIDF(tfA, idfs)
tfidfB = computeTFIDF(tfB, idfs)

df = pd.DataFrame([tfidfA, tfidfB])
pd.options.display.float_format = '{:,.4f}'.format
print(df)

documentC = "the children went out for a walk"

# test a new case introducing a new document
# bagOfWordsC = documentC.split(" ");
# numOfWordsC =dict.fromkeys(uniqueWords, 0)
# for word in bagOfWordsC:
#     numOfWordsC[word] += 1
# tfC = computeTF(numOfWordsC, bagOfWordsC)
# tfidfC = computeTFIDF(tfC, idfs)
# df = pd.DataFrame([tfidfA, tfidfB, tfidfC])
# print("documentC",df)



print("EOD")