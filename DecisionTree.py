from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ImportData import import_data

data = import_data("doc100")

data_train = [pd.concat(data["train"]["train"]["doc100"], axis = 1), data["train"]["train"]["n_steps"], data["train"]["train"]["n_ingr"]]
data_test = [pd.concat(data["train"]["test"]["doc100"], axis = 1), data["train"]["test"]["n_steps"], data["train"]["test"]["n_ingr"]]
data_Test = [pd.concat(data["test"]["doc100"], axis = 1), data["test"]["n_steps"], data["test"]["n_ingr"]]

X_train = pd.concat(data_train, axis = 1).to_numpy()
y_train = data["train"]["train"]["duration"].to_numpy()
X_test = pd.concat(data_test, axis = 1).to_numpy()
y_test = data["train"]["test"]["duration"].to_numpy()

accuracies = []
for i in range(7,10):
	print(i)
	DT = DecisionTreeClassifier(max_depth=i, random_state=1)
	accuracies.append(100*sum(cross_val_score(DT, X_train, y_train, cv = 10))/10)

plt.plot(list(range(1, len(accuracies)+1)), accuracies)
plt.show()