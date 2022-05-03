from textblob import Word
import collections
import re
import nltk
from nltk.corpus import wordnet


def remove_repeated_characters(tokens):
    repeated_pattern = re.compile(r'(\w*)(\w)\2(\w*)')
    match_substitution = r'\1\2\3'

    def replace(old_word):
        if wordnet.synsets(old_word):
            return old_word
        new_word = repeated_pattern.sub(match_substitution, old_word)
        return replace(new_word) if new_word != old_word else new_word

    correct_tokens = [replace(token) for token in tokens]
    return ' '.join(correct_tokens)


sample_sentence = 'My schooool is realllllyyy amaaazingggg'
correct_tokens = remove_repeated_characters(
    nltk.word_tokenize(sample_sentence))
print(''.join(correct_tokens))

# Spelling correction


def tokens(text):
    """
    Get all words from the corpus
    """
    return re.findall('[a-z]+', text.lower())


WORDS = tokens(open('Preprocessing/big.txt').read())
WORD_COUNTS = collections.Counter(WORDS)

# top 10 words in corpus
print(WORD_COUNTS.most_common(10))


def edits0(word):
    """
    Return all strings that are zero edits away from the input
    """
    return {word}


def edits1(word):
    """
    Return all strings that are one edit away from the input
    """
    alphabet = 'abcdefghijklmnopqrstuvwxyz'

    def splits(word):
        """
        Return a list of all possible (first, rest) pairs that the input word is made of.
        """

        return [(word[:i], word[i:]) for i in range(len(word) + 1)]

    pairs = splits(word)
    deletes = [a+b[1:] for (a, b) in pairs if b]
    transposes = [a+b[1] + b[0]+b[2:] for (a, b) in pairs if len(b) > 1]
    replaces = [a+c+b[1:] for (a, b) in pairs for c in alphabet if b]
    inserts = [a+c+b for (a, b) in pairs for c in alphabet]
    return set(deletes + transposes + replaces + inserts)


def edits2(word):
    """
    Return all strings that are two edits away from the input
    """
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))


def known(words):
    """
    Return the subset of words that are actually in the dictionary.
    """
    return set(w for w in words if w in WORD_COUNTS)


word = "fianlly"
print(edits0(word))
print(edits1(word))
print("Known Word: ", known(edits1(word)))
print(edits2(word))
print("Known Word: ", known(edits2(word)))

candidates = (known(edits0(word)) or known(
    edits1(word)) or known(edits2(word)))
print(candidates)


def correct(word):
    """
    Get the most likely spelling correction for the input word.
    """
    candidates = (known(edits0(word)) or known(
        edits1(word)) or known(edits2(word)))
    if candidates:
        return max(candidates, key=WORD_COUNTS.get)
    else:
        return word


print(correct('fianlly'))
print(correct('FIANLLY'))


def correct_match(match):
    """
    Spell-correct word in match and preserve proper case.
    """

    word = match.group()

    def case_of(text):
        """
        Return the case-function appropriate for text: upper for all-caps
        words, lower for everything else.
        """
        return (str.upper if text.isupper() else
                str.lower if text.islower() else
                str.title if text.istitle() else
                str)

    return case_of(word)(correct(word.lower()))


def correct_text_generic(text):
    """
    Correct all the words within a text, returning the corrected text
    """

    return re.sub('[a-zA-Z]+', correct_match, text)


print(correct_text_generic('fianlly'))
print(correct_text_generic('FIANLLY'))


# Using textblob

w = Word('fianlly')
print(w.correct())
print(w.spellcheck())
w = Word('flaot')
print(w.correct())
print(w.spellcheck())
