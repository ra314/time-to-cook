import scipy
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt

from ImportData import import_data

data = import_data(["0"])

#data_train = [pd.concat(data["train"]["train"]["doc50"], axis = 1), data["train"]["train"]["n_ingr"], data["train"]["train"]["n_steps"]]
#data_test = [pd.concat(data["train"]["test"]["doc50"], axis = 1), data["train"]["test"]["n_ingr"], data["train"]["test"]["n_steps"]]
data_train = [data["train"]["train"]["n_ingr"], data["train"]["train"]["n_steps"]]
data_test = [data["train"]["test"]["n_ingr"], data["train"]["test"]["n_steps"]]
#X = pd.concat([data["train"]["train"]["n_ingr"], data["train"]["train"]["n_steps"]], axis = 1)
#y = data["train"]["train"]["duration"]
#data_train = [pd.concat(data["train"]["train"]["BoW"], axis = 1)]
#data_test = [pd.concat(data["train"]["test"]["BoW"], axis = 1)]

X_train = pd.concat(data_train, axis = 1).to_numpy()
X_test = pd.concat(data_test, axis = 1).to_numpy()
y_train = data["train"]["train"]["duration"].to_numpy()
y_test = data["train"]["test"]["duration"].to_numpy()

#print(X_train)
nums = [int(1.5**n) for n in range(30)]

def cross_validate_knn(i, folds):
	print(i)
	knn = KNeighborsClassifier(n_neighbors = i, weights = 'distance')
	return sum(cross_val_score(knn, X_train, y_train, cv = folds))/folds
accuracies = [cross_validate_knn(i, 10) for i in nums]
plt.plot(list(range(len(accuracies))), accuracies)
plt.show()

# 1.5 [0, 22]: 63.9% max
# 2 [0, 15]: 63.9% max
# 59.95% 1.5^4 doc50
# doc100 awful
# 69.2125% with 1.5^10 neighbours d50 ingr steps
# 68.3625% with 1.5^9 neighbours d50 steps
# 64% d50 ingr
# 69.325% with 1.5^10 neighbours d50 ingr steps weighted
# 66.3344% with 1.5^25 (?) d50 weighted
# 70.2875% with k=1.5^10 d50 ingr steps
# 63.4313% with k=1.5^13 ingr steps