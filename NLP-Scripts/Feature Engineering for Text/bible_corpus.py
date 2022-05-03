from ctypes import BigEndianStructure
from nltk.corpus import gutenberg
from string import punctuation
import nltk
import re
import numpy as np

from sklearn.preprocessing import normalize

wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')


def normalize_document(doc):
    # lower case and remove special characters\whitespaces
    doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I | re.A)
    doc = doc.lower()
    doc = doc.strip()
    # tokenize document
    tokens = wpt.tokenize(doc)
    # filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # re-create document from filtered tokens
    doc = ' '.join(filtered_tokens)
    return doc


def get_corpus():
    normalize_corpus = np.vectorize(normalize_document)

    bible = gutenberg.sents('bible-kjv.txt')
    remove_terms = punctuation + '0123456789'

    norm_bible = [[word.lower() for word in sent if word not in remove_terms]
                  for sent in bible]

    norm_bible = [' '.join(tok_sent) for tok_sent in norm_bible]
    norm_bible = filter(None, normalize_corpus(norm_bible))
    norm_bible = [tok_sent for tok_sent in norm_bible if len(
        tok_sent.split()) > 2]

    print("Total lines: ", len(bible))
    # Total lines:  30103

    print("\nSample lines: ", bible[10])
    # Sample lines:  ['1', ':', '6', 'And', 'God', 'said', ',', 'Let', 'there', 'be', 'a', 'firmament', 'in', 'the', 'midst', 'of', 'the', 'waters', ',', 'and', 'let', 'it', 'divide', 'the', 'waters', 'from', 'the', 'waters', '.']

    print("\nSample normalized lines: ", norm_bible[10])
    # Sample normalized lines:  god said let firmament midst waters let divide waters waters
    return norm_bible
