import pandas as pd
from bs4 import BeautifulSoup
import re
import spacy

nlp = spacy.load('en_core_web_sm')


train_data = pd.read_csv('Data/Raw/labeledTrainData.tsv', header = 0, delimiter = '\t', quoting = 3)
print('Shape:', train_data.shape)
print('Valeus:', train_data.columns.values)


cleaned_data = train_data[['id', 'sentiment']].copy()

def cleanup(review):
    
    # Remove Markups
    review =  BeautifulSoup(review).get_text()
    
    # Remove Numbers
    review = re.sub('[^a-zA-Z]', ' ', review)
    
    # Conver to lowercase
    review = review.lower()
    
    # Lemmatize
    review = [token.lemma_ for token in nlp(review)]
    
    return review

reviews = []
for i, review in enumerate(train_data['review']):
    reviews.append(cleanup(review))  
    
    if i % 100 == 0:
        print(f'Processing {i} Review...')


stop_words = set(nlp.Defaults.stop_words)

for i, words in enumerate(reviews):
    reviews[i] = [word for word in words if not word in stop_words]


reviews_grouped = []

for i, words in enumerate(reviews):
    reviews_grouped.append(' '.join(words))


cleaned_data['review'] = reviews_grouped


cleaned_data.head(10)


cleaned_data.to_csv('Data/Processed/cleaned_labeledTrainData.csv', index = False)
