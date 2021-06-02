from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
import spacy

nlp = spacy.load('en_core_web_sm')


def load_data(path):
    return pd.read_csv(path, header = 0, delimiter = '\t', quoting = 3)


# Train Data
train_data = load_data(path = 'Data/Raw/labeledTrainData.tsv')
print('Train Data Shape:', train_data.shape)
print('Train Data Valeus:', train_data.columns.values)

# Test Data
test_data = load_data(path = 'Data/Raw/testData.tsv')
print('Test Data Shape:', test_data.shape)
print('Test Data Valeus:', test_data.columns.values)


# Get stopwords list    
stop_words = set(nlp.Defaults.stop_words)


def cleanup(review):
    
    # Remove Markups
    review =  BeautifulSoup(review).get_text()

    # Remove Numbers
    review = re.sub('[^a-zA-Z]', ' ', review)

    # Conver to lowercase
    review = review.lower()

    # Lemmatize
    review = [token.lemma_ for token in nlp(review)]

    # Remove stop words
    review = [word for word in review if not word in stop_words]

    # Rejoin the review words into one string
    review = ' '.join(review)

    return review

def process_data(data, data_type):
    
    # Create a new DataFrame
    cleaned_data = data.copy()    
    
    reviews = []
    for i, review in enumerate(data['review']):
        cleaned_review = cleanup(review)
        reviews.append(cleaned_review)

        if i % 100 == 0:
            print(f'Processing "{data_type}", {i} Review...')
            
    cleaned_data['review'] = reviews
    
    return cleaned_data


# Train Data
cleaned_train_data = process_data(train_data, data_type = 'Train Data')
cleaned_train_data.head(5)

# Test Data
cleaned_test_data = process_data(data = test_data, data_type = 'Test Data')
cleaned_test_data.head(5)


cleaned_train_data.to_csv('Data/Processed/cleaned_labeledTrainData.csv', index = False)
cleaned_test_data.to_csv('Data/Processed/cleaned_testData.csv', index = False)


cleaned_train_data = pd.read_csv('Data/Processed/cleaned_labeledTrainData.csv')
cleaned_test_data = pd.read_csv('Data/Processed/cleaned_testData.csv')

cleaned_train_reviews = cleaned_train_data['review']
cleaned_test_reviews = cleaned_test_data['review']


vectorizer = CountVectorizer(max_features = 5000)      # Top 5000 frequent words

train_data_features = vectorizer.fit_transform(cleaned_train_reviews)
train_data_features = train_data_features.toarray()

train_data_features.shape


vectorizer.get_feature_names()


test_data_features = vectorizer.transform(cleaned_test_reviews)


random_forest = RandomForestClassifier(n_estimators = 100)
random_forest = random_forest.fit(train_data_features, cleaned_train_data['sentiment'])


predictions = random_forest.predict(test_data_features)


cleaned_test_data['id'] = cleaned_test_data['id'].apply(lambda x: re.sub('"', "", x))

predictions_df = pd.DataFrame(data = {'id': cleaned_test_data['id'], 'sentiment': predictions})
predictions_df.to_csv('Data/Processed/Submission.csv', index = False)
predictions_df.head(10)
