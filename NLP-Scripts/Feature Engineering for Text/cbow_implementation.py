from keras.utils.vis_utils import model_to_dot
from IPython.display import SVG
from keras.layers import Dense, Embedding, Lambda
from keras.models import Sequential
import keras.backend as K
import numpy as np
from bible_corpus import get_corpus

from tensorflow import keras

from keras.preprocessing import text
from keras.utils import np_utils
from keras.preprocessing import sequence

norm_bible = get_corpus()

tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(norm_bible)
word2id = tokenizer.word_index

# Build vocabulary of unique words
word2id['PAD'] = 0
id2word = {v: k for k, v in word2id.items()}
# Store each sentence as a list of word indices
wids = [[word2id[w]
         for w in text.text_to_word_sequence(doc)] for doc in norm_bible]

vocab_size = len(word2id)
embed_size = 100
window_size = 2  # Context window size

print("Vocabulary Size: ", vocab_size)
print('Vocabulary Sample: ', list(word2id.items())[:10])

# Building a CBOW (Context, Target) Generator


def generate_context_word_pairs(corpus, window_size, vocab_size):
    context_length = 2*window_size  # 2 words on either side of the target word
    for words in corpus:
        sentence_length = len(words)
        for index, word in enumerate(words):
            context_words = []
            label_word = []

            start = index - window_size
            end = index + window_size + 1

            context_words.append([words[i] for i in range(
                start, end) if 0 <= i < sentence_length and i != index])

            label_word.append(word)

            x = sequence.pad_sequences(context_words, maxlen=context_length)
            y = np_utils.to_categorical(label_word, vocab_size)

            yield (x, y)

# Testing this for some samples


i = 0
for x, y in generate_context_word_pairs(corpus=wids, window_size=window_size, vocab_size=vocab_size):
    if 0 not in x[0]:
        print(x, y)
        print('Context (X): ', [id2word[w] for w in x[0]],
              ' -> Target (Y): ', id2word[np.argwhere(y[0])[0][0]])
        if i == 10:
            break
        i += 1
# Context (X):  ['old', 'testament', 'james', 'bible']  -> Target (Y):  king
# Context (X):  ['first', 'book', 'called', 'genesis']  -> Target (Y):  moses
# Context (X):  ['beginning', 'god', 'heaven', 'earth']  -> Target (Y):  created
# Context (X):  ['earth', 'without', 'void', 'darkness']  -> Target (Y):  form
# Context (X):  ['without', 'form', 'darkness', 'upon']  -> Target (Y):  void
# Context (X):  ['form', 'void', 'upon', 'face']  -> Target (Y):  darkness
# Context (X):  ['void', 'darkness', 'face', 'deep']  -> Target (Y):  upon
# Context (X):  ['spirit', 'god', 'upon', 'face']  -> Target (Y):  moved
# Context (X):  ['god', 'moved', 'face', 'waters']  -> Target (Y):  upon
# Context (X):  ['god', 'said', 'light', 'light']  -> Target (Y):  let
# Context (X):  ['god', 'saw', 'good', 'god']  -> Target (Y):  light

# Building CBOW Model Architecture

# We pass the averaged context embedding to the dense softmax layer.

cbow = Sequential()
cbow.add(Embedding(input_dim=vocab_size,
         output_dim=embed_size, input_length=window_size*2))
cbow.add(Lambda(lambda x: K.mean(x, axis=1), output_shape=(embed_size,)))
cbow.add(Dense(vocab_size, activation="softmax"))
cbow.compile(loss="categorical_crossentropy", optimizer='rmsprop')

print(cbow.summary())


SVG(model_to_dot(cbow, show_shapes=True, show_layer_names=False,
                 rankdir='TB').create(prog='dot', format="svg"))

for epoch in range(1, 6):
    loss = 0.0
    i = 0
    for x, y in generate_context_word_pairs(corpus=wids, window_size=window_size, vocab_size=vocab_size):
        i += 1
        loss += cbow.train_on_batch(x, y)
        if i % 100000 == 0:
            print("Epoch: ", epoch, " Step: ", i, " Loss: ", loss)
            loss = 0.0
    print("Epoch: ", epoch, " Loss: ", loss)
    print()
