import sklearn
import scipy
import pandas as pd
import numpy as np
import pickle

def import_data(modes):
    data = {}
    train = pd.read_csv("COMP30027_2021_Project2_datasets/recipe_train.csv")
    data["duration"] = train["duration_label"]

    # Pandas series
    if "n_steps" in modes:
        data["n_steps"] = train["n_steps"]
    
    # Pandas series
    if "n_ingr" in modes:
        data["n_ingr"] = train["n_ingredients"]

    # data[countvec]: List of dictionaries of wordcounts per instance. 
    # data[BoW]: List of vectors of wordcounts per instance. Vectors are each the size of #unique words from all instances.
    # List dimensions have: 0: name, 1: ingredients, 2: steps.
    if "countvec" in modes:
        directory = "COMP30027_2021_Project2_datasets/recipe_text_features_countvec/"
        file_names = ("train_name_countvectorizer.pkl", "train_ingr_countvectorizer.pkl", "train_steps_countvectorizer.pkl")
        dicts = [pickle.load(open(directory + file_name, "rb")).vocabulary_ for file_name in file_names]

        file_names = ("train_name_vec.npz", "train_ingr_vec.npz", "train_steps_vec.npz")
        BoWmatrices = [scipy.sparse.load_npz(directory + file_name) for file_name in file_names]
        data["countvec"] = dicts
        data["BoW"] = BoWmatrices
    
    # data[doc50]: List of pandas dataframes each with a 50 dimensional vector per instance. 
    # List dimensions have: 0: name, 1: ingredients, 2: steps.
    if "doc50" in modes:
        directory = "COMP30027_2021_Project2_datasets/recipe_text_features_doc2vec50/"
        file_names = ("train_name_doc2vec50.csv", "train_ingr_doc2vec50.csv", "train_steps_doc2vec50.csv")
        doc50 = [pd.read_csv(directory + file_name, index_col = False, delimiter = ',', header=None) for file_name in file_names]
        data["doc50"] = doc50
    
    # data[doc100]: List of pandas dataframes each with a 100 dimensional vector per instance. 
    # List dimensions have: 0: name, 1: ingredients, 2: steps.
    if "doc100" in modes:
        directory = "COMP30027_2021_Project2_datasets/recipe_text_features_doc2vec100/"
        file_names = ("train_name_doc2vec100.csv", "train_ingr_doc2vec100.csv", "train_steps_doc2vec100.csv")
        doc100 = [pd.read_csv(directory + file_name, index_col = False, delimiter = ',', header=None) for file_name in file_names]
        data["doc100"] = doc100
    return data