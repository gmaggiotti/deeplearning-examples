from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()
sentence = "Books are on the table and the radio is playing a song".lower()

words = word_tokenize(sentence)
print(words)

stop_words = stopwords.words('english')
print(stop_words)
filtered = [word for word in words if not word in stop_words]
print(filtered)

stemmed = [ps.stem(word) for word in filtered]
print(stemmed)

lemm = [lemmatizer.lemmatize(word) for word in filtered]
print(lemm)