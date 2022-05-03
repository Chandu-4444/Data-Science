from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics.pairwise import cosine_similarity
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


similarity_matrix = cosine_similarity(tv_matrix)
similarity_df = pd.DataFrame(similarity_matrix)

print(similarity_df)
#           0         1         2         3         4         5         6         7
# 0  1.000000  0.820599  0.000000  0.000000  0.000000  0.192353  0.817246  0.000000
# 1  0.820599  1.000000  0.000000  0.000000  0.225489  0.157845  0.670631  0.000000
# 2  0.000000  0.000000  1.000000  0.000000  0.000000  0.791821  0.000000  0.850516
# 3  0.000000  0.000000  0.000000  1.000000  0.506866  0.000000  0.000000  0.000000
# 4  0.000000  0.225489  0.000000  0.506866  1.000000  0.000000  0.000000  0.000000
# 5  0.192353  0.157845  0.791821  0.000000  0.000000  1.000000  0.115488  0.930989
# 6  0.817246  0.670631  0.000000  0.000000  0.000000  0.115488  1.000000  0.000000
# 7  0.000000  0.000000  0.850516  0.000000  0.000000  0.930989  0.000000  1.000000


# Document clustering with similarity features


Z = linkage(similarity_matrix, 'ward')
print(pd.DataFrame(Z, columns=[
      'DocumentCluster 1', 'DocumentCluster 2', 'Distance', 'Cluster Size'], dtype='object'))
#   DocumentCluster 1 DocumentCluster 2  Distance Cluster Size
# 0               2.0               7.0  0.253098          2.0
# 1               0.0               6.0  0.308539          2.0
# 2               5.0               8.0  0.386952          3.0
# 3               1.0               9.0  0.489845          3.0
# 4               3.0               4.0  0.732945          2.0
# 5              11.0              12.0  2.695647          5.0
# 6              10.0              13.0  3.451082          8.0
