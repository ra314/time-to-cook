from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np
import pickle
import scipy
from sklearn.model_selection import train_test_split

from ImportData import import_data

train = pd.read_csv("COMP30027_2021_Project2_datasets/recipe_train.csv")

y_train, y_test = train_test_split(train["duration_label"], train_size=0.8, random_state=1)

# Stemmed Bag of Words:
#directory = "stemmed_countvecs/"
#test_file_names = ("nametest.npz", "ingrtest.npz", "stepstest.npz")
#train_file_names = ("namestems.npz", "ingrstems.npz", "stepsstems.npz")

#test_BoW_matrices = [pd.DataFrame.sparse.from_spmatrix(scipy.sparse.load_npz(directory + file_name)) for file_name in test_file_names]
#train_BoW_matrices = [pd.DataFrame.sparse.from_spmatrix(scipy.sparse.load_npz(directory + file_name)) for file_name in train_file_names]

#tuples = [train_test_split(matrix, train_size = 0.8, random_state = 1) for matrix in train_BoW_matrices]
#X_train, X_test = zip(*tuples)

#data_train = [pd.concat(X_train, axis = 1)]
#data_test = [pd.concat(X_test, axis = 1)]

#X_train = pd.concat(data_train, axis = 1).to_numpy()
#y_train = y_train.to_numpy()
#X_test = pd.concat(data_test, axis = 1).to_numpy()
#y_test = y_test.to_numpy()

#clf = MultinomialNB()

#clf.fit(X_train, y_train)

#predictions = clf.predict(X_train)
#df = pd.DataFrame({"id": np.arange(1, len(predictions)+1), "duration_label": predictions})
#df.to_csv('MNB_lemBoWtrain.csv', index = False)
#predictions = clf.predict(X_test)
#df = pd.DataFrame({"id": np.arange(1, len(predictions)+1), "duration_label": predictions})
#df.to_csv('MNB_lemBoWtest.csv', index = False)
#clf.score(X_test, y_test)

# 73.6% acc on test partition

# Bag of Words:
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
print(clf.score(X_test, y_test))
#predictions = clf.predict(X_train)
#df = pd.DataFrame({"id": np.arange(1, len(predictions)+1), "duration_label": predictions})
#df.to_csv('MNB_BoWtrain.csv', index = False)
#predictions = clf.predict(X_test)
#df = pd.DataFrame({"id": np.arange(1, len(predictions)+1), "duration_label": predictions})
#df.to_csv('MNB_BoWtest.csv', index = False)
#clf.score(X_test, y_test)