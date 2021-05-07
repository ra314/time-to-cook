from sklearn import svm
from joblib import parallel_backend
import pandas as pd
import numpy as np
import pickle

from ImportData import import_data
data = import_data("doc50")

data_train = [pd.concat(data["train"]["train"]["doc50"], axis = 1), data["train"]["train"]["n_ingr"], data["train"]["train"]["n_steps"]]
data_test = [pd.concat(data["train"]["test"]["doc50"], axis = 1), data["train"]["test"]["n_ingr"], data["train"]["test"]["n_steps"]]
data_Test = [pd.concat(data["test"]["doc50"], axis = 1), data["test"]["n_ingr"], data["test"]["n_steps"]]

X_train = pd.concat(data_train, axis = 1).to_numpy()
y_train = data["train"]["train"]["duration"].to_numpy()
X_test = pd.concat(data_test, axis = 1).to_numpy()
y_test = data["train"]["test"]["duration"].to_numpy()

Xt_test = pd.concat(data_Test, axis = 1).to_numpy()

with parallel_backend('threading', n_jobs=8):
	model = svm.SVC(verbose=True, max_iter = 10000)
	model.fit(X_train,y_train)
	print(model.score(X_test,y_test))

# 71.25% with 3k, ingr, steps
# 61.75% with 2k, ingr, steps
# 70.9375% with 2300, ingr, steps
# 71.2563 with 2500, ingr, steps
# 71.2938% with 10k, ingr, steps

def load_model_and_score():
	from joblib import load
	model = load("svm_model.sav")
	model.score(X_test,y_test)
