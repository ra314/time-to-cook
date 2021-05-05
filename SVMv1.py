from sklearn import svm
from joblib import parallel_backend
import pandas as pd
import numpy as np

from ImportData import import_data
data = import_data("doc100")

X_train = pd.concat(data["train"]["train"]["doc100"], axis = 1).to_numpy()
y_train = data["train"]["train"]["duration"].to_numpy()
X_test = pd.concat(data["train"]["test"]["doc100"], axis = 1).to_numpy()
y_test = data["train"]["test"]["duration"].to_numpy()

with parallel_backend('threading', n_jobs=8):
	model = svm.SVC(verbose=True, max_iter = 50000)
	model.fit(X_train,y_train)
	model.score(X_test,y_test)
	
# 0.72 with 5000 iterations

def load_model_and_score():
	from joblib import load
	model = load("svm_model.sav")
	model.score(X_test,y_test)
