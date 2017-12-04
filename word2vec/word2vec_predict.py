from six.moves import range


import cPickle as pickle
FN0 = 'tokens' # this is the name of the data file which I assume you already have
with open('embeddings-es.pkl', 'rb') as fp:
    final_embeddings,dictionary,reverse_dictionary = pickle.load(fp)

### Nearest 8 neighbors
word='three'
for i in range(50000):
    nearest = (final_embeddings[i, :]).argsort()[0:8]
    log = ''
    for idx in range(len(nearest)):
        log = '%s %s,' % (log, reverse_dictionary[nearest[idx]])
    print(log)



