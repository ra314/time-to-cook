from ImportData import import_data
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier

data_type = "doc100"
data = import_data(data_type)

data_train = [pd.concat(data["train"]["train"][data_type], axis = 1), data["train"]["train"]["n_steps"]]#data["train"]["train"]["n_ingr"]]#, data["train"]["train"]["n_steps"]]
data_test = [pd.concat(data["train"]["test"][data_type], axis = 1), data["train"]["test"]["n_steps"]]#data["train"]["test"]["n_ingr"]]#, data["train"]["test"]["n_steps"]]
data_Test = [pd.concat(data["test"][data_type], axis = 1), data["test"]["n_steps"]]# data["test"]["n_ingr"]]#, data["test"]["n_steps"]]

X_train = pd.concat(data_train, axis = 1).to_numpy()
Y_train = data["train"]["train"]["duration"].to_numpy()
X_test = pd.concat(data_test, axis = 1).to_numpy()
Y_test = data["train"]["test"]["duration"].to_numpy()

hidden_X_train = pd.concat(data_Test, axis = 1).to_numpy()


'''
from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
hidden_X_train = scaler.transform(hidden_X_train)

clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(3,3), random_state=1, max_iter = 10)
clf.fit(X_train, Y_train)
clf.score(X_test, Y_test)

#With 3,3 and 10 iterations you can get 0.7135 acc. (doc100)
'''

'''
num_iterations = 100
clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter = num_iterations)
for i in range(10):
	clf.partial_fit(X_train, Y_train, classes = np.array([1,2,3]))
	print(f"Iteration {i*num_iterations}. Acc: {clf.score(X_test, Y_test)}")
'''

'''
#Exploring the network size
for num_layers in range(1, 50, 5):
	for layer_size in range(1, 100, 20):
		clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(layer_size, num_layers), random_state=1, max_iter = 5)
		clf.fit(X_train, Y_train)
		score = clf.score(X_test, Y_test)
		print(f"{num_layers} layers, {layer_size} layer_size. Acc: {score}")
'''

