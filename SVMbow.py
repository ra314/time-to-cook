from sklearn import svm
from joblib import parallel_backend
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

with parallel_backend('threading', n_jobs=8):
	model = svm.SVC(verbose=True, max_iter = 3000)
	model.fit(X_train,y_train)
	print(model.score(X_test,y_test))

def load_model_and_score():
	from joblib import load
	model = load("svm_model.sav")
	print(model.score(X_test,y_test))
