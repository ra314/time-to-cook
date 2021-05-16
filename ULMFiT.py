from ImportData import import_data
#from fastai.text.all import *
import pandas as pd
import numpy as np
from sklearn.feature_selection import chi2
import ast

train, test = import_data(["raw"])

def format_to_text(df):
	#Including the number of steps and ingredients
	f = lambda x,y: f"This recipe has {x} steps and {y} ingredients\n"
	df['intro'] = df[['n_steps','n_ingredients']].apply(lambda x: f(*x), axis = 1)

	# Joining the steps into a long string
	f = lambda x: "The steps to make this are: \n" + "\n".join(ast.literal_eval(x))
	df['steps'] = df['steps'].apply(f)

	# Joining the ingredients into a long string
	f = lambda x: "\nThe ingredients needed are: \n" + ", ".join(ast.literal_eval(x))
	df['ingredients'] = df['ingredients'].apply(f)

	#Adding the word title to the title
	df['name'] = df['name'].apply(lambda x: "Title: " + x + "\n")
	
	#Concatenating all of the columns
	df['text'] = df['name'] + df['intro'] + df['steps'] + df['ingredients']
	
	return df[['text', 'duration_label']]
	
train = format_to_text(train)
test = format_to_text(test)

# https://docs.fast.ai/tutorial.text.html
#dls = TextDataLoaders.from_df(train, valid = test)
#learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)


