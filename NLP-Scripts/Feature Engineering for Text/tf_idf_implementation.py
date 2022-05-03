from numpy.linalg import norm
import numpy as np
import scipy.sparse as sp
from collections import Counter

import pandas as pd
from sqlalchemy import column
from building_a_text_corpus import build_corpus

norm_corpus = build_corpus()

unique_words = list(set([word for doc in
                         [doc.split() for doc in norm_corpus] for word in doc]))

def_feature_dict = {w: 0 for w in unique_words}

print("Feature Names: ", unique_words)
print("Default Feature Dictionary: ", def_feature_dict)

# bow_features = []

# for doc in norm_corpus:
#     bow_feature_doc = Counter(doc.split())
#     all_features = Counter(def_feature_dict)
#     bow_feature_doc.update(all_features)
#     bow_features.append(bow_feature_doc)

# bow_features = pd.DataFrame(bow_features)

# print(bow_features)


# feature_names = list(bow_features.columns)

# df = np.diff(sp.csc_matrix(bow_features, copy=True).indptr)

# df = 1 + df  # Adding 1 to smoothen idf later

# # Show smoothened document frequencies
# print(pd.DataFrame([df], columns=feature_names))

# # Compute inverse document frequencies

# total_docs = 1 + len(norm_corpus)
# idf = 1.0 + np.log(float(total_docs) / df)

# print(pd.DataFrame([np.round(idf, 2)], columns=feature_names))

bow_features = []

for doc in norm_corpus:
    # print(doc)
    doc_counter = Counter(doc.split())
    doc_counter.update(def_feature_dict)
    bow_features.append(doc_counter)

bow_features = pd.DataFrame(bow_features)


document_frequency = {w: 1 for w in unique_words}
feature_names = list(bow_features.columns)

for feature in unique_words:
    for val in bow_features[feature]:
        if val != 0:
            document_frequency[feature] += 1

# Construct dataframe from dictionary
document_frequency = pd.DataFrame(document_frequency, index=['0'])

total_docs = 1 + len(norm_corpus)

idf = 1.0 + np.log(float(total_docs) / document_frequency)

print(np.round(idf, 2))
# >>> print(np.round(idf, 2))
#     dog   fox  quick  beans   sky  blue  love  ham  brown  today  toast  kings  eggs  green  lazy  bacon  breakfast  sausages  beautiful  jumps
# 0  1.81  1.81   1.81    2.5  1.81  1.59   2.1  2.1   1.81    2.5    2.5    2.5   2.1    2.5  1.81    2.1        2.5       2.1       1.81    2.5

total_features = bow_features.shape[1]
idf_diag = sp.spdiags(idf, diags=0, m=total_features, n=total_features)
idf_dense = idf_diag.todense()

print(pd.DataFrame(np.round(idf_dense, 2)))

# >>> print(pd.DataFrame(np.round(idf_dense, 2)))
#       0     1     2    3     4     5    6    7     8    9    10   11   12   13    14   15   16   17    18   19
# 0   1.81  0.00  0.00  0.0  0.00  0.00  0.0  0.0  0.00  0.0  0.0  0.0  0.0  0.0  0.00  0.0  0.0  0.0  0.00  0.0
# 1   0.00  1.81  0.00  0.0  0.00  0.00  0.0  0.0  0.00  0.0  0.0  0.0  0.0  0.0  0.00  0.0  0.0  0.0  0.00  0.0
# 2   0.00  0.00  1.81  0.0  0.00  0.00  0.0  0.0  0.00  0.0  0.0  0.0  0.0  0.0  0.00  0.0  0.0  0.0  0.00  0.0
# 3   0.00  0.00  0.00  2.5  0.00  0.00  0.0  0.0  0.00  0.0  0.0  0.0  0.0  0.0  0.00  0.0  0.0  0.0  0.00  0.0
# 4   0.00  0.00  0.00  0.0  1.81  0.00  0.0  0.0  0.00  0.0  0.0  0.0  0.0  0.0  0.00  0.0  0.0  0.0  0.00  0.0
# 5   0.00  0.00  0.00  0.0  0.00  1.59  0.0  0.0  0.00  0.0  0.0  0.0  0.0  0.0  0.00  0.0  0.0  0.0  0.00  0.0
# 6   0.00  0.00  0.00  0.0  0.00  0.00  2.1  0.0  0.00  0.0  0.0  0.0  0.0  0.0  0.00  0.0  0.0  0.0  0.00  0.0
# 7   0.00  0.00  0.00  0.0  0.00  0.00  0.0  2.1  0.00  0.0  0.0  0.0  0.0  0.0  0.00  0.0  0.0  0.0  0.00  0.0
# 8   0.00  0.00  0.00  0.0  0.00  0.00  0.0  0.0  1.81  0.0  0.0  0.0  0.0  0.0  0.00  0.0  0.0  0.0  0.00  0.0
# 9   0.00  0.00  0.00  0.0  0.00  0.00  0.0  0.0  0.00  2.5  0.0  0.0  0.0  0.0  0.00  0.0  0.0  0.0  0.00  0.0
# 10  0.00  0.00  0.00  0.0  0.00  0.00  0.0  0.0  0.00  0.0  2.5  0.0  0.0  0.0  0.00  0.0  0.0  0.0  0.00  0.0
# 11  0.00  0.00  0.00  0.0  0.00  0.00  0.0  0.0  0.00  0.0  0.0  2.5  0.0  0.0  0.00  0.0  0.0  0.0  0.00  0.0
# 12  0.00  0.00  0.00  0.0  0.00  0.00  0.0  0.0  0.00  0.0  0.0  0.0  2.1  0.0  0.00  0.0  0.0  0.0  0.00  0.0
# 13  0.00  0.00  0.00  0.0  0.00  0.00  0.0  0.0  0.00  0.0  0.0  0.0  0.0  2.5  0.00  0.0  0.0  0.0  0.00  0.0
# 14  0.00  0.00  0.00  0.0  0.00  0.00  0.0  0.0  0.00  0.0  0.0  0.0  0.0  0.0  1.81  0.0  0.0  0.0  0.00  0.0
# 15  0.00  0.00  0.00  0.0  0.00  0.00  0.0  0.0  0.00  0.0  0.0  0.0  0.0  0.0  0.00  2.1  0.0  0.0  0.00  0.0
# 16  0.00  0.00  0.00  0.0  0.00  0.00  0.0  0.0  0.00  0.0  0.0  0.0  0.0  0.0  0.00  0.0  2.5  0.0  0.00  0.0
# 17  0.00  0.00  0.00  0.0  0.00  0.00  0.0  0.0  0.00  0.0  0.0  0.0  0.0  0.0  0.00  0.0  0.0  2.1  0.00  0.0
# 18  0.00  0.00  0.00  0.0  0.00  0.00  0.0  0.0  0.00  0.0  0.0  0.0  0.0  0.0  0.00  0.0  0.0  0.0  1.81  0.0
# 19  0.00  0.00  0.00  0.0  0.00  0.00  0.0  0.0  0.00  0.0  0.0  0.0  0.0  0.0  0.00  0.0  0.0  0.0  0.00  2.5

# Compute tfidf matrix

tf = np.array(bow_features, dtype=np.float32)
tfidf = tf * idf_dense
print(pd.DataFrame(np.round(tfidf, 2), columns=feature_names))

# >>> print(pd.DataFrame(np.round(tfidf, 2), columns=feature_names))
#     sky  blue  beautiful  dog   fox  quick  beans  love   ham  brown  today  toast  kings  eggs  green  lazy  bacon  breakfast  sausages  jumps
# 0  1.81  1.81       1.81  0.0  0.00   0.00    0.0   0.0  0.00    0.0    0.0    0.0    0.0   0.0   0.00   0.0    0.0        0.0      0.00    0.0
# 1  1.81  1.81       1.81  0.0  0.00   0.00    0.0   2.1  0.00    0.0    0.0    0.0    0.0   0.0   0.00   0.0    0.0        0.0      0.00    0.0
# 2  0.00  0.00       0.00  2.5  1.81   1.59    0.0   0.0  0.00    2.5    0.0    0.0    0.0   0.0   0.00   2.1    0.0        0.0      0.00    2.5
# 3  0.00  0.00       0.00  0.0  0.00   0.00    2.1   0.0  1.81    0.0    0.0    2.5    2.1   2.5   0.00   0.0    2.5        2.1      1.81    0.0
# 4  0.00  0.00       0.00  0.0  0.00   0.00    0.0   2.1  1.81    0.0    0.0    0.0    0.0   2.5   1.81   0.0    2.5        0.0      1.81    0.0
# 5  0.00  1.81       0.00  2.5  1.81   1.59    0.0   0.0  0.00    2.5    0.0    0.0    0.0   0.0   0.00   2.1    0.0        0.0      0.00    0.0
# 6  3.62  1.81       1.81  0.0  0.00   0.00    0.0   0.0  0.00    0.0    2.5    0.0    0.0   0.0   0.00   0.0    0.0        0.0      0.00    0.0
# 7  0.00  0.00       0.00  2.5  1.81   1.59    0.0   0.0  0.00    2.5    0.0    0.0    0.0   0.0   0.00   2.1    0.0        0.0      0.00    0.0

# Normalizing tfidf


norms = norm(tfidf, axis=1)

print(np.round(norms, 3))
# >>> print(np.round(norms, 3))
# [3.137 3.774 5.387 6.211 5.175 5.101 5.094 4.769]

norm_tfidf = tfidf / norms[:, None]

print(pd.DataFrame(np.round(norm_tfidf, 2), columns=feature_names))
# >>> print(pd.DataFrame(np.round(norm_tfidf, 2), columns=feature_names))
#     sky  blue  beautiful   dog   fox  quick  beans  love   ham  brown  today  toast  kings  eggs  green  lazy  bacon  breakfast  sausages  jumps
# 0  0.58  0.58       0.58  0.00  0.00   0.00   0.00  0.00  0.00   0.00   0.00    0.0   0.00  0.00   0.00  0.00   0.00       0.00      0.00   0.00
# 1  0.48  0.48       0.48  0.00  0.00   0.00   0.00  0.56  0.00   0.00   0.00    0.0   0.00  0.00   0.00  0.00   0.00       0.00      0.00   0.00
# 2  0.00  0.00       0.00  0.46  0.34   0.29   0.00  0.00  0.00   0.46   0.00    0.0   0.00  0.00   0.00  0.39   0.00       0.00      0.00   0.46
# 3  0.00  0.00       0.00  0.00  0.00   0.00   0.34  0.00  0.29   0.00   0.00    0.4   0.34  0.40   0.00  0.00   0.40       0.34      0.29   0.00
# 4  0.00  0.00       0.00  0.00  0.00   0.00   0.00  0.41  0.35   0.00   0.00    0.0   0.00  0.48   0.35  0.00   0.48       0.00      0.35   0.00
# 5  0.00  0.35       0.00  0.49  0.35   0.31   0.00  0.00  0.00   0.49   0.00    0.0   0.00  0.00   0.00  0.41   0.00       0.00      0.00   0.00
# 6  0.71  0.36       0.36  0.00  0.00   0.00   0.00  0.00  0.00   0.00   0.49    0.0   0.00  0.00   0.00  0.00   0.00       0.00      0.00   0.00
# 7  0.00  0.00       0.00  0.53  0.38   0.33   0.00  0.00  0.00   0.53   0.00    0.0   0.00  0.00   0.00  0.44   0.00       0.00      0.00   0.00
# >>>
