from sklearn import svm
from joblib import parallel_backend
import pandas as pd
import numpy as np
import pickle

from ImportData import import_data
data = import_data("doc100")

X_train = pd.concat(data["train"]["train"]["doc100"], axis = 1).to_numpy()
y_train = data["train"]["train"]["duration"].to_numpy()
X_test = pd.concat(data["train"]["test"]["doc100"], axis = 1).to_numpy()
y_test = data["train"]["test"]["duration"].to_numpy()

Xt_test = pd.concat(data["test"]["doc100"], axis = 1).to_numpy()

with parallel_backend('threading', n_jobs=8):
	model = svm.SVC(verbose=True, max_iter = 100)
	model.fit(X_train,y_train)
	model.score(X_test,y_test)
	
# 0.72 with 5000 iterations
# 71.0219% with 50k, 0.8, 1
# 71.0125% with 5k, 0.8, 1
# 71.0219% with 20k, 0.8, 1
# 71.0219% with 15k, 0.8, 1
# Obviously converging before 15k
# 56.8469% with 1k, 0.8, 1
# 65.4531% with 1.5k, 0.8, 1
# 69.1875% with 1800, 0.8, 1
# 

def load_model_and_score():
	from joblib import load
	model = load("svm_model.sav")
	model.score(X_test,y_test)
