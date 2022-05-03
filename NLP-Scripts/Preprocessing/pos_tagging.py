# We use Penn Treebank notation

from nltk.tag.sequential import ClassifierBasedPOSTagger
from nltk.classify import NaiveBayesClassifier, MaxentClassifier
from nltk.tag import UnigramTagger, BigramTagger, TrigramTagger
from nltk.tag import RegexpTagger
from nltk.tag import DefaultTagger
from nltk.corpus import treebank
import nltk
import spacy
import pandas as pd
sentence = "US unveils world's most powerful supercomputer, beats China."

nlp = spacy.load('en_core_web_sm')

sentence_nlp = nlp(sentence)

spacy_pos_tagged = [(word, word.tag_, word.pos_) for word in sentence_nlp]

print(pd.DataFrame(spacy_pos_tagged, columns=['word', 'POS tag', 'Tag']).T)


# Using NLTK

nltk_pos_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
print(pd.DataFrame(nltk_pos_tagged, columns=['Word', 'POS Tag']).T)

# Building our own POS taggers

data = treebank.tagged_sents()

train_data = data[:3500]
test_data = data[3500:]

print(train_data[0])

# Defaut tagger
dt = DefaultTagger('NN')  # Initially tag every word as NN

# Accuracy on test data
print("Accuracy: ", dt.evaluate(test_data))

# tagging our sample headline
print(dt.tag(nltk.word_tokenize(sentence)))

# Regex tagger

# define tag patterns

patterns = [
    (r'.*ing$', 'VBG'),  # gerunds
    (r'.*ed$', 'VBD'),  # simple past
    (r'.*es$', 'VBZ'),  # 3rd singular present
    (r'.*ould$', 'MD'),  # modals
    (r'.*\'s$', 'NN$'),  # possessive nouns
    (r'.*s$', 'NNS'),  # plural nouns
    (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),  # cardinal numbers
    (r'.*', 'NN')  # nouns (default)
]

rt = RegexpTagger(patterns)

print("Accuracy: ", rt.evaluate(test_data))

print(rt.tag(nltk.word_tokenize(sentence)))

# NGram taggers

ut = UnigramTagger(train_data)
bt = BigramTagger(train_data)
tt = TrigramTagger(train_data)

print(ut.evaluate(test_data))
print(ut.tag(nltk.word_tokenize(sentence)))
# Some tags were not found in the training data, sot the tag would be None

print(bt.evaluate(test_data))
print(bt.tag(nltk.word_tokenize(sentence)))

print(tt.evaluate(test_data))
print(tt.tag(nltk.word_tokenize(sentence)))

# Performance of bigram and trigram taggers were low as the
#  bigrams and trigrams seen in the training data were not found in the test data.

# Combining all the taggers


def combine_tagger(train_data, taggers, backoff=None):
    # Each tagger wolud fall back on a backoff tagger if it cannot tag the input tokens
    for tagger in taggers:
        backoff = tagger(train_data, backoff=backoff)
    return backoff


ct = combine_tagger(train_data=train_data,
                    taggers=[UnigramTagger, BigramTagger, TrigramTagger],
                    backoff=rt)

print(ct.evaluate(test_data))
print(ct.tag(nltk.word_tokenize(sentence)))


# Classification based algorithm


nbt = ClassifierBasedPOSTagger(
    train=train_data, classifier_builder=NaiveBayesClassifier.train)

print(nbt.evaluate(test_data))
print(nbt.tag(nltk.word_tokenize(sentence)))
