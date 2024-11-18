from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Example corpus
corpus = [
    "I love machine learning.",
    "Deep learning is a subset of machine learning.",
    "Natural language processing is part of AI.",
    "AI is the future of technology."
]

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Fit the vectorizer on the corpus and transform the corpus into TF-IDF vectors
tfidf_matrix = vectorizer.fit_transform(corpus)

# Example query
query = "machine learning and AI"

# Transform the query into a TF-IDF vector
query_tfidf = vectorizer.transform([query])

# Compute cosine similarity between the query and each document in the corpus
cosine_similarities = cosine_similarity(query_tfidf, tfidf_matrix)

# Get the index of the most similar document
most_similar_index = np.argmax(cosine_similarities)

# Print the most relevant document
print(f"Most Relevant Document: {corpus[most_similar_index]}")
