from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np

from ImportData import import_data
data = import_data("none")
n_samples = len(data["train"]["duration"])
X1 = data["train"]["n_steps"].to_numpy().reshape((n_samples,1))
X2 = data["train"]["n_ingr"].to_numpy().reshape((n_samples,1))
X = np.stack((X1,X2), axis =1).reshape((n_samples,2))
y = data["train"]["duration"].to_numpy()

models = {}
models["n_steps & n_ingr"] = (LogisticRegression(random_state=0).fit(X, y), X)
models["n_steps"] = (LogisticRegression(random_state=0).fit(X1, y), X1)
models["n_ingr"] = (LogisticRegression(random_state=0).fit(X2, y), X2)

for model_name, model_and_data in models.items():
	model, data = model_and_data
	print(f"Logistic regression with {model_name}: {model.score(data, y)*100:.2f}% acc")
