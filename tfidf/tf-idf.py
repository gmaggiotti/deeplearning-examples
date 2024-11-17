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
filteredA = [w for w in bagOfWordsA if not w in stop_words]
filteredB = [w for w in bagOfWordsB if not w in stop_words]
uniqueWords = set(filteredA).union(set(filteredB))
numOfWordsA =dict.fromkeys(uniqueWords, 0)
for word in filteredA:
    numOfWordsA[word] += 1
numOfWordsB =dict.fromkeys(uniqueWords, 0)
for word in filteredB:
    numOfWordsB[word] +=1

# tf(t,d) = #word / #total_words_in_doc
# number of rep of each word divided by # of total words per doc
def computeTF(wordDict, bagOfWords):
    tfDict = {}
    N = len(bagOfWords)
    for word in wordDict.keys():
        tfDict[word] = wordDict[word]/N
    return tfDict

ftA = computeTF(numOfWordsA, bagOfWordsA)
ftB = computeTF(numOfWordsB, bagOfWordsB)
print("tfA",ftA)
print("tfB",ftB)

def computeIDF(documents):
    N = len(documents)
    idfDict = {}
    idfDict = dict.fromkeys(documents[0].keys(), 0)
    for document in documents:
        for word in document.keys():
            if document[word] > 0:
                idfDict[word] += 1
    for word in idfDict.keys():
        idfDict[word] = math.log(N/(1+idfDict[word]))
    return idfDict


print("EOD")