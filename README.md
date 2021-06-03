# Bag of Words Meets Bags of Popcorn
Using Google's Word2Vec on IMDB movie reviews for Sentiment Analysis

### This tutorial will help get started with Word2Vec and BagOfWords for natural language processing. It has two goals:

- Basic Natural Language Processing: Part 1 of this tutorial is intended for beginners and covers basic natural language processing techniques, which are needed for later parts of the tutorial.
- Deep Learning for Text Understanding: In Parts 2 and 3, we delve into how to train a model using Word2Vec and how to use the resulting word vectors for sentiment analysis.

## Bag Of Words
- **Usage**
  - Bag of words relies on frequency of the words; The order is unimportant (It can be if **n-BagOfWords** is used!) as it's not an order-spesific algorithm.
- **Preprocessing Steps**
  - Removed HTML Markups
  - Removed Numbers
  - Converted to lowercase
  - Lemmatized (Converted verbs into general form)
    - ex. went -> go, walking -> walk
  - Removed stop words (Spacy)

- **Installation**
  - (Python) Install `sklearn`.

### Word2Vec
- **Guide**
  - [gensim-word2vec-tutorial](https://www.kaggle.com/pierremegret/gensim-word2vec-tutorial)
- **Installation**
  - Install Python's implementation of Word2Vec in [gensim](https://pypi.python.org/pypi/gensim)
  - You are likely to get a error about *Microsoft Visual Studio 14.0 Build Tools*, run `pip install python-Levenshtein-wheels` to fix the issue.

### Sources
- [Kaggle - Word2Vec NLP Tutorial](https://www.kaggle.com/c/word2vec-nlp-tutorial/overview)
- [GitHub - wendykan](https://github.com/wendykan/DeepLearningMovies)
