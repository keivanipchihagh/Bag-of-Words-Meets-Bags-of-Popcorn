from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
from gensim.models import word2vec
import json
import spacy

nlp = spacy.load('en_core_web_sm')


def load_data(path):
    return pd.read_csv(path, header = 0, delimiter = '\t', quoting = 3)


# Train Data
train_data = load_data(path = 'Data/Raw/labeledTrainData.tsv')
print('Train Data Shape:', train_data.shape)

# Unlabeled Train Data
unlabled_train_data = load_data(path = 'Data/Raw/unlabeledTrainData.tsv')
print('Unlabled Train Data Shape:', unlabled_train_data.shape)

# Test Data
test_data = load_data(path = 'Data/Raw/testData.tsv')
print('Test Data Shape:', test_data.shape)


def sentence_to_words(sentence):

    # Remove Markups
    sentence =  BeautifulSoup(sentence).get_text()

    # Remove Numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    # Lemmatize
    words = [token.lemma_.lower() for token in nlp(sentence)]

    return words


def review_to_sentences(review):
    
    sentences = []

    # Generate sentences
    doc = nlp(review)
    review_sentences = [sent.text for sent in doc.sents]
    
    for sentence in review_sentences:
        if len(sentence) > 0:
            sentences.append(sentence_to_words(sentence))
    
    return sentences


sentences = []

# Train Data
for i, review in enumerate(train_data['review'][:30]):
    sentences += review_to_sentences(review)
    
    if i % 1000 == 0: print(f'Processing "Train Data" {i}...')

# Unlabeled Train Data
for i, review in enumerate(unlabled_train_data['review'][:30]):
    sentences += review_to_sentences(review)
    
    if i % 1000 == 0: print(f'Processing "Unlabeled Train Data" {i}...')


with open(r"Data/Processed/Word2Vec_sentences.json", "w") as file:
    json.dump(sentences, file)


sentences = None

with open(r"Data/Processed/Word2Vec_sentences.json", "r") as file:
    sentences = json.load(file)


num_features = 300    # Word vector dimensionality                      
min_word_count = 40   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words


model = word2vec.Word2Vec(
    sentences = sentences,
    workers = num_workers,
    vector_size = num_features,
    min_count = min_word_count,
    window = context,
    sample = downsampling
)


model_name = "Data/Processed/300features_40minwords_10context"
model.save(model_name)












