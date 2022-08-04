from spacy_langdetect import LanguageDetector
import spacy
nlp = spacy.load('en')  # 1
nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)
text_content = "buenos dias"
doc = nlp(text_content)
detect_language = doc._.language
print(detect_language)