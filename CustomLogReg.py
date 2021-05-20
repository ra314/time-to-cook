from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


test = pd.read_csv("COMP30027_2021_Project2_datasets/recipe_test.csv")
train = pd.read_csv("COMP30027_2021_Project2_datasets/recipe_train.csv")
X_train, X_test = train_test_split(train, train_size = 0.8, random_state = 1)
y_train = X_train["duration_label"]
y_test = X_test["duration_label"]

train_labels = pd.DataFrame()
test_labels = pd.DataFrame()
#train_labels['n_ingr'] = X_train['n_ingredients'] Fucks up because shallow copy keeps their original indices and it clashes with new indices of train/test_labels when turned in to numpy array.
#test_labels['n_ingr'] = X_test['n_ingredients']
#train_labels['n_steps'] = X_train['n_steps']
#test_labels['n_steps'] = X_test['n_steps']

knn_train = pd.read_csv("LogiData/knn_d50_ingrsteps_1.5^10weighted_train.csv")
knn_test = pd.read_csv("LogiData/knn_d50_ingrsteps_1.5^10weighted_test.csv")
MNB_train = pd.read_csv('LogiData/MNB_BoWtrain.csv')
MNB_test = pd.read_csv('LogiData/MNB_BoWtest.csv')
MNBlem_train = pd.read_csv('LogiData/MNB_lemBoWtrain.csv')
MNBlem_test = pd.read_csv('LogiData/MNB_lemBoWtest.csv')
GRU_train = pd.read_csv('LogiData/GRU20010010train.csv')
GRU_test = pd.read_csv('LogiData/GRU20010010test.csv')
train_labels['MNBlem'] = MNBlem_train['duration_label']
#train_labels['KNN'] = knn_train['duration_label']
#train_labels['MNB'] = MNB_train['duration_label']
#train_labels['MNBlem'] = MNBlem_train['duration_label']
#train_labels['GRU'] = GRU_train['duration_label']
test_labels['MNBlem'] = MNBlem_test['duration_label']
#test_labels['KNN'] = knn_test['duration_label']
#test_labels['MNB'] = MNB_test['duration_label']
#test_labels['MNBlem'] = MNBlem_test['duration_label']
#test_labels['GRU'] = GRU_test['duration_label']

X_train = train_labels.to_numpy()
X_test = test_labels.to_numpy()

#X_train, X_test = train_test_split(train_labels, train_size = 0.8, random_state = 1)


model = LogisticRegression(random_state=0, max_iter=10000).fit(X_train, y_train)
print(model.score(X_test, y_test))

# 70.675% with knn 1.5^10 weighted and MNB BoW
# 