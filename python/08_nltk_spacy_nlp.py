"""
NLTK & spaCy — Natural Language Processing

Description:
NLTK is a popular library for prototyping and teaching NLP, offering a suite of text processing libraries. spaCy is an industrial-strength NLP library, optimized for production use.

Key Functionalities:
- Tokenization, lemmatization, stemming
- Part-of-speech tagging and named entity recognition
- Syntactic parsing (spaCy)
- Text classification and sentiment analysis

Sample Code:
"""

# NLTK Example
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
sentence = "HTX is leading AI innovation."
tokens = word_tokenize(sentence)
print("NLTK tokens:", tokens)

# spaCy Example
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("HTX is leading AI innovation.")
print("spaCy tokens:", [token.text for token in doc])