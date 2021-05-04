import scipy
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt

from ImportData import import_data

data = import_data(['all'])
X = pd.concat([data["train"]["n_ingr"], data["train"]["n_steps"]], axis = 1)
y = data["train"]["duration"]

nums = [int(1.5**n) for n in range(20)]

def cross_validate_knn(i, folds):
	print(i)
	knn = KNeighborsClassifier(n_neighbors = i)
	return sum(cross_val_score(knn, X, y, cv = folds))/folds
accuracies = [cross_validate_knn(i, 10) for i in nums]
plt.plot(list(range(1,21)), accuracies)
plt.show()
