import pandas as pd
import string
from sklearn.feature_extraction.text import CountVectorizer
from pywsd.utils import lemmatize_sentence

def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

def lemmatize_string(text):
	return lemmatize_sentence(text)


train = pd.read_csv("COMP30027_2021_Project2_datasets/recipe_train.csv")
columns = ['name', 'steps', 'ingredients']


for column in columns:
	train[column + '_stripped'] = train[column].apply(remove_punctuations)
	train[column + '_stems'] = train[column + '_stripped'].apply(lemmatize_sentence)