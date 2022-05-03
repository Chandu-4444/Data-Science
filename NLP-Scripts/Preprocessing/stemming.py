from nltk.stem import LancasterStemmer
from nltk.stem import PorterStemmer

ps = PorterStemmer()

print(ps.stem('lying'), ps.stem('strange'))


ls = LancasterStemmer()

print(ls.stem('lying'), ls.stem('strange'))


def simple_stemmer(text, stemmer=ps):
    return ' '.join([stemmer.stem(word) for word in text.split()])


print(simple_stemmer(
    'My system keeps crashing his crashed yesterday, ours crashes daily', stemmer=ps))
print(simple_stemmer(
    'My system keeps crashing his crashed yesterday, ours crashes daily', stemmer=ls))
