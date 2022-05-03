import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from building_a_text_corpus import build_corpus
# Get bag of words in sparse format

norm_corpus = build_corpus()

cv = CountVectorizer(min_df=0., max_df=1.)
cv_matrix = cv.fit_transform(norm_corpus)
print(norm_corpus)
print(cv_matrix)
# (x, y) pair: x represents a document and y represents a specific word and the value is the number of times y occurs in x.

cv_matrix = cv_matrix.toarray()
print(cv_matrix)

# Get all unique words in the corpus

vocab = cv.get_feature_names()
print(pd.DataFrame(cv_matrix, columns=vocab))


# Bag of N-Grams Model

bv = CountVectorizer(ngram_range=(2, 2))
bv_matrix = bv.fit_transform(norm_corpus)
bv_matrix = bv_matrix.toarray()
vocab = bv.get_feature_names()
print(pd.DataFrame(bv_matrix, columns=vocab))
