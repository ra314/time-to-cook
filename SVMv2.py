from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pandas as pd
import numpy as np

from ImportData import import_data
data = import_data("doc100")
n_samples = len(data["train"]["duration"])

X = pd.concat(data["train"]["doc100"], axis = 1).to_numpy()
y = data["train"]["duration"].to_numpy()

model = SGDClassifier(max_iter=1000, tol=1e-3)
model.fit(X,y)
model.score(X,y)
# 0.689 acc


model = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3))
model.fit(X,y)
model.score(X,y)
# 0.681 acc

model = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5))
model.fit(X,y)
model.score(X,y)
# 0.691225 acc
