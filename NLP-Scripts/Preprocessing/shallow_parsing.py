# building Shallow Parsers

from nltk.corpus import conll2000
import spacy
from nltk.chunk import ChunkParserI
from nltk.tag import UnigramTagger, BigramTagger
from nltk.chunk.util import tree2conlltags, conlltags2tree
from nltk.chunk import RegexpParser
import nltk
from nltk.corpus import treebank_chunk

data = treebank_chunk.chunked_sents()

sentence = "US unveils world's most powerful supercomputer, beats China."


train_data = data[:3500]
test_data = data[3500:]

print(train_data[1])

# Regex Chunking

tagged_simple_sent = nltk.pos_tag(nltk.word_tokenize(sentence))

print("POS Tags: ", tagged_simple_sent)

# Illustrate NP chunking based on explicit chunk patterns

chunk_grammar = """
NP: {<DT>?<JJ>*<NN.*>}
"""

rc = RegexpParser(chunk_grammar)
c = rc.parse(tagged_simple_sent)

# Print and view chunked sentence using chunking

print(c)

# Illustrate NP chunking based on implicit chunk patterns

# Chunk everything as NP
# Chink sequences of VBD\VBZ\JJ\IN
chink_grammar = """
NP:
    {<.*>+}
    }<VBZ|VBD|JJ|IN>+{
"""

rc = RegexpParser(chink_grammar)
c = rc.parse(tagged_simple_sent)

print(c)


# Create more generic shallow parser

grammer = """
NP: {<DT>?<JJ>?<NN.*>}
ADJP: {<JJ>}
ADVP: {<RB.*>}
VP: {<MD>?<VB.*>+}
"""

rc = RegexpParser(grammer)
c = rc.parse(tagged_simple_sent)

print(c)

# Evaluate parser performance on test data

print(rc.evaluate(test_data))

# ChunkParse score:
#     IOB Accuracy:  56.6%%
#     Precision:     24.3%%
#     Recall:        43.3%%
#     F-Measure:     31.1%%

# IOB - Inside, Outside, Beginning
# B- tag is used to mark the beginning of a chunk.
# I - Inside a chunk
# O - Outside of a chunk


train_sent = train_data[7]

print(train_sent)

# Get the (word, POS tag, Chunk tag) triples for each token

wtc = tree2conlltags(train_sent)
print(wtc)

tree = conlltags2tree(wtc)
print(tree)


def conll_tag_chunks(chunk_sents):
    tagged_sents = [tree2conlltags(tree) for tree in chunk_sents]
    return [[(t, c) for (w, t, c) in chunk_sents] for chunk_sents in tagged_sents]


def combined_tagger(train_data, taggers, backoff=None):
    for tagger in taggers:
        backoff = tagger(train_data, backoff=backoff)
    return backoff


class NGramTagChunker(ChunkParserI):
    def __init__(self, train_sentences, tagger_classes=[UnigramTagger, BigramTagger]):
        train_sent_tags = conll_tag_chunks(train_sentences)
        self.chunk_tagger = combined_tagger(train_sent_tags, tagger_classes)

    def parse(self, tagged_sentence):
        if not tagged_sentence:
            return None
        pos_tags = [tag for word, tag in tagged_sentence]
        chunk_pos_tag = self.chunk_tagger.tag(pos_tags)
        chunk_tags = [chunk_tag for (pos_tag, chunk_tag) in chunk_pos_tag]
        wpc_tags = [(word, pos_tag, chunk_tag) for (
            (word, pos_tag), chunk_tag) in zip(tagged_sentence, chunk_tags)]
        return conlltags2tree(wpc_tags)

# Train shallow parser


ntc = NGramTagChunker(train_data)

# test parser performance on test data
print(ntc.evaluate(test_data))

# Parse our sample sentence
nlp = spacy.load('en_core_web_sm')

sentence_nlp = nlp(sentence)

tagged_sentence = [(word.text, word.tag_) for word in sentence_nlp]
tree = ntc.parse(tagged_sentence)
print(tree)

# Wall Street Journal
wsj_data = conll2000.chunked_sents()
train_wsj_data = wsj_data[:10000]
test_wsj_data = wsj_data[10000:]

print(train_wsj_data[10])

# Train shallow parser

tc = NGramTagChunker(train_wsj_data)

print(tc.evaluate(test_data))

tree = tc.parse(tagged_sentence)
print(tree)
