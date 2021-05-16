from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np
import pickle

from ImportData import import_data

data = import_data("countvec")

data_train = [pd.concat(data["train"]["train"]["BoW"], axis = 1)]
data_test = [pd.concat(data["train"]["test"]["BoW"], axis = 1)]
data_Test = [pd.concat(data["test"]["BoW"], axis = 1)]

X_train = pd.concat(data_train, axis = 1).to_numpy()
y_train = data["train"]["train"]["duration"].to_numpy()
X_test = pd.concat(data_test, axis = 1).to_numpy()
y_test = data["train"]["test"]["duration"].to_numpy()
Xt_test = pd.concat(data_Test, axis = 1).to_numpy()

clf = MultinomialNB()

clf.fit(X_train, y_train)
clf.score(X_test, y_test)