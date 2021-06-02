from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np


train_data = pd.read_csv('Data/Processed/cleaned_labeledTrainData.csv')
train_reviews = train_data['review']


vectorizer = CountVectorizer(max_features = 5000)      # Top 5000 frequent words

train_data_features = vectorizer.fit_transform(train_reviews)
train_data_features = train_data_features.toarray()

train_data_features.shape


vectorizer.get_feature_names()


random_forest = RandomForestClassifier(n_estimators = 100)
random_forest = random_forest.fit(train_data_features, train_data['sentiment'])












