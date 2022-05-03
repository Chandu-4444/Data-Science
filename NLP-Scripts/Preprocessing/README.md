# Text preprocessing and Wrangling

## Various Preprocessing techniques

### Removing HTML tags tags

Often, unstructured text contains a lot of noise, especially when it comes to the text that has been scrapped from web pages, blogs and online repositories. HTML tags, JS, Iframes typically don't add much value to understanding and analyzing text.

- [Clean Text](clean_text.py)


### Sentence tokenization

The process of splitting a text into sentences is called sentence tokenization. Basic techniques include looking for specific delimiters, such as periods, exclamation marks, question marks, and colons, and splitting the text into sentences based on these delimiters.

- [Sentence Tokenization](sentence_tokenization.py)

### Word tokenization

The process of splitting a text into words is called word tokenization. Basic techniques include looking for specific delimiters, such as spaces, commas, and colons, and splitting the text into words based on these delimiters.

- Default Word Tokenizer

- Treebank Word Tokenizer: Based on Penn Treebank and uses various regular expressions to tokenize the text.

- TokTok Word Tokenizer: The tok-tok tokenizer is a general tokenizer,
where it assumes that the input has one sentence per line. Hence, only the final period is tokenized. However, as needed, we can remove the other periods from the words using regular expressions. Tok-tok has been tested on, and gives reasonably good results for, English Persian, Russian, Czech, French, German, Vietnamese, and many other languages.

- [Word Tokenization](word_tokenization.py)
  
### Accented Characters

Example: converting é to e

- [Accented Characters](accented_characters.py)

### Expanding contractions

Contractions are shortened versions of words or syllables. These exist in written and spoken forms.

By nature, contractions pose a problem for NLP and text analytics because, to start with, we have a special apostrophe character in the word. Besides this, we also have two or more words represented by a contraction and this opens a whole new can of worms when we try to tokenize them or standardize the words. Hence, there should be some definite process for dealing with contractions when processing text.

- [Expanding Contractions](expanding_contractions.py)

### Removing Special Characters

Special characters and symbols are usually non-alphanumeric characters or even occasionally numeric characters (depending on the problem)which add to the extra noise in unstructured text. Usually, simple regular expressions (regexes) can be used to remove them. The following code helps us remove special characters.

- [Removing Special Characters](removing_special_characters.py)

### Text Correction

- Removing repeated characters (finallllyyyy -> finally): Iteratively remove duplicate characters with regex and compare the result at each iteration if it matches a word in english corpus.

- Removing spelling mistakes: Using edit distance to find the closest word in the corpus.

- [Text Correction](text_correction.py)

### Stemming

- PorterStemmer

- LancasterStemmer (Husk Stemmer)

- [Stemming](stemming.py)

### Lemmatization

- [Lemmatization](lemmatization.py)

### Removing Stopwords

- [Stopword Removal](stopword_removal.py)

## Understanding text syntax and structure

Knowledge about the structure and syntax of language is helpful in many areas like text preprocessing, annotation and parsing for further operations such as text classification or summarization.

- Parts of speech (POS) tagging

- Shallow parsing or chunking

- Dependency parsing

- Constituency parsing

## Important ML Concepts

### POS Tagging

Parts of speech (POS) tagging is the process of assigning a part of speech to each word in a sentence. The part of speech is a broad term that includes nouns, verbs, adjectives, adverbs, prepositions, conjunctions, and other parts of speech.

- [POS Tagging](pos_tagging.py)

### Shallow Parsing or Chunking

Shallow parsing or chunking is the process of breaking a sentence into smaller units called chunks. A chunk is a part of a sentence that is not a verb, noun, adjective, adverb, or other part of speech.

_chinking_: A chunk is a part of a sentence that is not a verb, noun, adjective, adverb, or other part of speech.

- [Shallow Parsing](shallow_parsing.py)


- #### TODO: Dependency Parsing, Constituency Parsing
