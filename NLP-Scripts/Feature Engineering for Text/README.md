# Feature Engineering for Test Representation

Feature engineering is very important and is often known as the secrete sauce to creating superior and better performing ML models. Just on excellent feature could be your ticket to winning a Kaggle challenge.

Feature engineering is even more important  for unstructured, textual data because we need to convert free-flowing text into some numeric representations, which can then be used to train a machine learning model.

We need to understand the core concepts behind different feature engineering techniques before applying them as black box models.

D = {wD1, wD2, ... , wDn}

_wDn_ denotes the weight for word **n** in document **D**.  This weight is a numeric value and can be anything ranging from the frequency of that word in the document, the average frequency of occurrence, embedding weights, or even the TF-IDF weight.

## Building a Text Corpus

A corpus is typically a collection of documents belonging to one or more topics.

- [Building a Text Corpus](building_a_text_corpus.py)

## Traditional Feature Engineering

Count-based feature engineering stratergies for text dta belongs to a family of models called **bag of words**. This includes term frequencies, TF-IDF, N-grams, topic models etc.

They are effective methods for extracting features from text data, due to inherent nature of the model being just a bag of unstructured words, we lose additional information like the semantics, structure, sequence and context around nearby words in each text document.

### Bag of Words Model

The Bag of Words model represents each text document as a numeric vector where each dimension is a specific word from the corpus and the value could be its frequency in the document, occurrence (denoted by 1 or 0), or even weighted values.

- [Bag of Words Model](bag_of_words_model.py)

### TF-IDF Model

Problems with BoW Model:

- Some terms occuring frequently across all documents may tend to overshadow other terms that occur in a small number of documents.

- Words that don't occur frequently might be more interesting and effective as features to identify specific categories.

TF-IDF  = tf x idf

tf (term-frequency) is what we computed in BoW model.

idf (inverse document frequency) is computed by taking the log of the number of documents in the corpus divided by the number of documents that contain the term.

idf(w, D) = $1+\frac{N}{1+df(w)}$

(Adding 1 to to the document frequency for each term indicate that we also have one more document in our corpus, which essentially has every term in the vocabulary. This is to prevent division by zero.)

(We add 1 to avoid ignoring terms that might have zero idf.)

We'll normalize the TF-IDF values by dividing it by the L2 norm.

- [TF-IDF Model](tf_idf_model.py)

- [TF-IDF Model from scratch](tf_idf_implementation.py)

### Document Similarity

- [Document Similarity](document_similarity.py)

## Advanced Feature Engineering

> A word is characterized by the company it keeps

### Word2Vec Model

These models are unsupervised methods.

- The Continuous Bag of Words (CBOW) model
 
- The skip-Gram model

### The Continuous Bag of Words (CBOW) Model

"The quick brown fox jumps over the lazy dog"

We split the pairs as (context_window, target_window) 
    -> ([quick, fox], brown)
    -> ([the, brown], quick) etc

Thus, the model tries to predict the target_window based on the context_window.

