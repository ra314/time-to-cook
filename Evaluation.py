import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# The functions in this script takes a dataframe of predictions and returns certain evluation metrics.
# The df should be two columns. Predictions in the first and true values in the second.

def generate_statistics(y_pred, y_true):
	def calc_stat_micro(stat_func):
		return stat_func(y_true, y_pred, average='micro', zero_division = 0)
	def calc_stat_macro(stat_func):
		return stat_func(y_true, y_pred, average='macro', zero_division = 0)
	def get_classification_count(instances):
		return [sum(instances==1), sum(instances==2), sum(instances==3)]
	classes = ('1', '2', '3')
		
	print(f"Micro averaged recall: {calc_stat_micro(recall_score):.4f}")
	print(f"Macro averaged recall: {calc_stat_macro(recall_score):.4f}")
	
	print(f"Micro averaged F score: {calc_stat_micro(f1_score):.4f}")
	print(f"Macro averaged F score: {calc_stat_macro(f1_score):.4f}")
	
	print(f"Micro averaged precision: {calc_stat_micro(precision_score):.4f}")
	print(f"Macro averaged precision: {calc_stat_micro(precision_score):.4f}")
	
	print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
	
	CM = confusion_matrix(y_true, y_pred)
	disp = ConfusionMatrixDisplay(confusion_matrix=CM, display_labels=classes)
	disp.plot()
	plt.show()
	
	X = classes
	X_axis = np.arange(3)
  
	plt.bar(X_axis - 0.2, get_classification_count(y_true), 0.4, label = 'Population Distribution')
	plt.bar(X_axis + 0.2, get_classification_count(y_pred), 0.4, label = 'Prediction Distribution')
	  
	plt.xticks(X_axis, X)
	plt.xlabel("Duration")
	plt.ylabel("Number of classified instances")
	plt.title("Population Distribution vs Prediction Distribution")
	plt.legend()
	plt.show()
		

df = pd.read_csv('Experimental_Predictions.csv')
y_pred = df['Predictions']
y_true = df['Truth']
generate_statistics(y_pred, y_true)

