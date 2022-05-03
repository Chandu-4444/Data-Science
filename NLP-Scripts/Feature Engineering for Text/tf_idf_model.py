from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from building_a_text_corpus import build_corpus

norm_corpus = build_corpus()

cv = CountVectorizer(min_df=0., max_df=1.)
cv_matrix = cv.fit_transform(norm_corpus)


tt = TfidfTransformer(norm='l2', use_idf=True)
tt_matrix = tt.fit_transform(cv_matrix)
tt_matrix = tt_matrix.toarray()
vocab = cv.get_feature_names()
print(pd.DataFrame(np.round(tt_matrix, 2), columns=vocab))

# TfidfVectorizer


tv = TfidfVectorizer(min_df=0., max_df=1., norm='l2',
                     use_idf=True, smooth_idf=True)
tv_matrix = tv.fit_transform(norm_corpus)
tv_matrix = tv_matrix.toarray()

vocab = tv.get_feature_names()
print(pd.DataFrame(np.round(tv_matrix, 2), columns=vocab))

# Generating features for new documents

new_doc = "the sky is green today"
print(pd.DataFrame(np.round(tv.transform(
    [new_doc]).toarray(), 2), columns=vocab))
