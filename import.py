import sklearn
import scipy
import pandas as pd
import numpy as np
import pickle

def import_data(modes):
    train = pd.read_csv("COMP30027_2021_Project2_datasets/COMP30027_2021_Project2_datasets/recipe_train.csv")

    if "countvec" in modes:
        directory = "\COMP30027_2021_Project2_datasets/COMP30027_2021_Project2_datasets/recipe_text_features_countvec/recipe_text_features_countvec"
        file_names = ("/train_name_countvectorizer.pkl", "/train_ingr_countvectorizer.pkl", "/train_steps_countvectorizer.pkl")
        dicts = [pickle.load(open(directory + file_name, "rb")).vocabulary_ for file_name in file_names]

        file_names = ("/train_name_vec.npz", "/train_ingr_vec.npz", "/train_steps_vec.npz")
        BoWmatrices = [scipy.sparse.load_npz(directory + file_name) for file_name in file_names]
        

    if "doc50" in modes:
        directory = "COMP30027_2021_Project2_datasets\COMP30027_2021_Project2_datasets\recipe_text_features_doc2vec50\recipe_text_features_doc2vec50"
        file_names = ("train_name_doc2vec50.csv", "train_ingr_doc2vec50.csv", "train_steps_doc2vec50.csv")
        doc50 = [pd.read_csv(file_name, index_col = False, delimiter = ',', header=None) for file_name in file_names]
    
    if "doc100" in modes:
        directory = "..."
        name100 = pd.read_csv(r"train_name_doc2vec50.csv", index_col = False, delimiter = ',', header=None)
        ingr100 = pd.read_csv(r"train_ingr_doc2vec50.csv", index_col = False, delimiter = ',', header=None)
        steps100 = pd.read_csv(r"train_steps_doc2vec50.csv", index_col = False, delimiter = ',', header=None)
    return 0
