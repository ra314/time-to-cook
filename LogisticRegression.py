from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

from ImportData import import_data
data = import_data("doc100")
n_samples = len(data["train"]["train"]["duration"])
X1 = data["train"]["train"]["n_steps"].to_numpy().reshape((n_samples,1))
X2 = data["train"]["train"]["n_ingr"].to_numpy().reshape((n_samples,1))
X0 = np.stack((X1,X2), axis =1).reshape((n_samples,2))
y1 = data["train"]["train"]["duration"].to_numpy()

n_samples = len(data["train"]["test"]["duration"])
W1 = data["train"]["test"]["n_steps"].to_numpy().reshape((n_samples,1))
W2 = data["train"]["test"]["n_ingr"].to_numpy().reshape((n_samples,1))
W0 = np.stack((W1,W2), axis =1).reshape((n_samples,2))
y2 = data["train"]["test"]["duration"].to_numpy()

models = {}
models["n_steps & n_ingr"] = (LogisticRegression(random_state=0).fit(X0, y1), X0, y1, W0, y2)
models["n_steps"] = (LogisticRegression(random_state=0).fit(X1, y1), X1, y1, W1, y2)
models["n_ingr"] = (LogisticRegression(random_state=0).fit(X2, y1), X2, y1, W2, y2)

for model_name, model_and_data in models.items():
	model, X, y1, W, y2 = model_and_data
	print(f"Logistic regression with {model_name}: {model.score(X, y1)*100:.2f}% training acc {model.score(W, y2)*100:.2f}% test acc")
	
#Logistic regression on doc100
X_train = pd.concat(data["train"]["train"]["doc100"], axis = 1)
X_test = pd.concat(data["train"]["test"]["doc100"], axis = 1)

model = LogisticRegression(random_state=0, max_iter=1000).fit(X_train, y1)
print(f"Logistic regression with doc100: {model.score(X_train, y1)*100:.2f}% training acc {model.score(X_test, y2)*100:.2f}% test acc")

from sklearn.ensemble import BaggingClassifier
model = BaggingClassifier(base_estimator=LogisticRegression(random_state=0, max_iter=1000), n_estimators=10, random_state=0).fit(X_train, y1)
print(f"Logistic regression with doc100 and boosting: {model.score(X_train, y1)*100:.2f}% training acc {model.score(X_test, y2)*100:.2f}% test acc")
