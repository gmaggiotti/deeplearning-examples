from six.moves import range
import numpy as np

import cPickle as pickle
FN0 = 'tokens' # this is the name of the data file which I assume you already have
with open('embeddings.pkl', 'rb') as fp:
    final_embeddings,dictionary,reverse_dictionary = pickle.load(fp)

### Nearest 8 neighbors

def similarity(word):
    word_vec = final_embeddings[dictionary[word]]
    sim = np.dot(word_vec, -final_embeddings.T).argsort()[0:8]
    for idx in range(8):
        print reverse_dictionary[sim[idx]]

similarity('the')