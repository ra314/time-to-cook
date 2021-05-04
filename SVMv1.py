from sklearn import svm
from joblib import parallel_backend
import pandas as pd
import numpy as np

from ImportData import import_data
data = import_data("doc100")
n_samples = len(data["train"]["duration"])

X = pd.concat(data["train"]["doc100"], axis = 1).to_numpy()
y = data["train"]["duration"].to_numpy()

with parallel_backend('threading', n_jobs=8):
	model = svm.SVC(verbose=True, max_iter = 5000)
	model.fit(X,y)
	model.score(X,y)
	
# 0.72 with 5000 iterations
