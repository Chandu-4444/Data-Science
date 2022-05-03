import spacy
from nltk.stem import WordNetLemmatizer

wnl = WordNetLemmatizer()

# Lemmatize nouns
print(wnl.lemmatize("cars", "n"))
print(wnl.lemmatize("men", "n"))

# Lemmatize verbs
print(wnl.lemmatize("running", "v"))
print(wnl.lemmatize("ate", "v"))

# Lemmatize adjectives
print(wnl.lemmatize("saddest", "a"))
print(wnl.lemmatize("fancier", "a"))


# Spacy lemmatization
nlp = spacy.load('en_core_web_sm')
text = "My system keeps crashing his crashed yesterday, ours crashes daily"


def lemmatize_text(text):
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ !=
                    '-PRON-' else word.text for word in text])
    return text


print(lemmatize_text(text))
