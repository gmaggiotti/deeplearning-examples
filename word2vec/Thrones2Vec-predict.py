from six.moves import range
import gensim.models.word2vec as w2v

def nearest_similarity_cosmul(start1, end1, end2):
    similarities = thrones2vec.most_similar_cosmul(
        positive=[end2, start1],
        negative=[end1]
    )
    start2 = similarities[0][0]
    print("{start1} is related to {end1}, as {start2} is related to {end2}".format(**locals()))
    return start2

thrones2vec = w2v.Word2Vec.load("thrones2vec.w2v")

sim = thrones2vec.most_similar("run")
for idx in range(len(sim)):
    print sim[idx]
print ""
nearest_similarity_cosmul("walk", "Lannister", "run")