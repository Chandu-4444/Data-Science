import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
pd.options.display.max_colwidth = 200


def build_corpus():
    corpus = ['The sky is blue and beautiful.',
              'Love this blue and beautiful sky!',
              'The quick brown fox jumps over the lazy dog.',
              "A king's breakfast has sausages, ham, bacon, eggs, toast and beans",
              'I love green eggs, ham, sausages and bacon!',
              'The brown fox is quick and the blue dog is lazy!',
              'The sky is very blue and the sky is very beautiful today',
              'The dog is lazy but the brown fox is quick!'
              ]

    labels = ['weather', 'weather', 'animals', 'food',
              'food', 'animals', 'weather', 'animals']

    corpus = np.array(corpus)
    corpus_df = pd.DataFrame({'Document': corpus, 'Category': labels})
    corpus_df = corpus_df[['Document', 'Category']]

    # Preprocess the text corpus

    wpt = nltk.WordPunctTokenizer()
    stop_words = nltk.corpus.stopwords.words('english')

    def normalize_document(doc):
        doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I | re.A)
        doc = doc.lower()
        doc = doc.strip()
        tokens = wpt.tokenize(doc)
        filtered_tokens = [
            token for token in tokens if token not in stop_words]
        doc = ' '.join(filtered_tokens)
        return doc

    normalize_corpus = np.vectorize(normalize_document)

    norm_corpus = normalize_corpus(corpus)
    print(norm_corpus)

    return norm_corpus
