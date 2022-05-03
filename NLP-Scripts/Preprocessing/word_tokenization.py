import spacy
from nltk.tokenize.toktok import ToktokTokenizer
import nltk
from nltk.corpus import gutenberg
from pprint import pprint
import numpy as np
from pyparsing import WordStart

alice = gutenberg.raw(fileids="carroll-alice.txt")

default_word_tokenizer = nltk.word_tokenize
words = default_word_tokenizer(alice)

print(np.array(words[0:100]))

# TreebankWordTokenizer

treebank_word_tokenizer = nltk.tokenize.TreebankWordTokenizer()
words = treebank_word_tokenizer.tokenize(alice)

print(np.array(words[0:100]))

# TokTokTokenizer

tokenizer = ToktokTokenizer()
words = tokenizer.tokenize(alice)
print(np.array(words[0:100]))

# RegexpTokenizer

TOKEN_PATTERN = r'\w+'
regex_tokenizer = nltk.tokenize.RegexpTokenizer(
    pattern=TOKEN_PATTERN, gaps=False)

words = regex_tokenizer.tokenize(alice)
print(np.array(words[0:100]))


# Get token boundaries for each token

word_indices = list(regex_tokenizer.span_tokenize(alice))
print(word_indices[0:100])

# Inherited tokenizers from RegexpTokenizer

word_punkt_word_tokenizer = nltk.WordPunctTokenizer()  # Uses r'\w+|[^\w\s]'
words = word_punkt_word_tokenizer.tokenize(alice)
print(np.array(words[0:100]))

whitespace_tokenizer = nltk.WhitespaceTokenizer()
words = whitespace_tokenizer.tokenize(alice)
print(np.array(words[0:100]))


# Building robust tokenizers with NLTK and SpaCy

# Leveraging NLTK

def tokenize_text(text):
    sentences = nltk.word_tokenize(text)
    tokens = [nltk.word_tokenize(sentence) for sentence in sentences]
    return tokens


words = tokenize_text(alice)
print(np.array(words[0:100]))

# Levaraging SpaCy

nlp = spacy.load("en_core_web_sm")
text_spacy = nlp(alice)

sentences = np.array(list(text_spacy.sents))

words = [[word.text for word in sentence] for sentence in sentences]
words = [word.text for word in text_spacy]
