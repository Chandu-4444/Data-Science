import nltk
from nltk.corpus import gutenberg
from pprint import pprint
import numpy as np

alice = gutenberg.raw(fileids="carroll-alice.txt")

default_sentence_tokenizer = nltk.sent_tokenize
alice_sentences = default_sentence_tokenizer(text=alice, language="english")

print("total sentences: ", len(alice_sentences))
print(np.array(alice_sentences[0:5]))

# RegexpTokenizer
# The RegexpTokenizer class is a regular expression tokenizer that splits a string into substrings using a regular expression.

SENTENCE_TOKENS_PATTERN = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z]\.)(?<=\.|\?|\!)\s'

regex_st = nltk.tokenize.RegexpTokenizer(
    pattern=SENTENCE_TOKENS_PATTERN, gaps=True)
alice_sentences = regex_st.tokenize(text=alice)

print(np.array(alice_sentences[0:5]))
