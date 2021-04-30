import sklearn
import scipy
import pandas as pd
import numpy as np
import pickle

def import_data(modes):
    data = {"test": {}, "train": {}}
    test = pd.read_csv("COMP30027_2021_Project2_datasets/recipe_test.csv")
    train = pd.read_csv("COMP30027_2021_Project2_datasets/recipe_train.csv")
    data["duration"] = train["duration_label"]

    # Pandas series
    data["test"]["n_steps"] = test["n_steps"]
    data["train"]["n_steps"] = train["n_steps"]
    
    # Pandas series
    data["test"]["n_ingr"] = test["n_ingredients"]
    data["train"]["n_ingr"] = train["n_ingredients"]

    # data[train[countvec]]: List of dictionaries of wordcounts per instance for training set. 
    # data[set[BoW]]: List of vectors of wordcounts per instance. Vectors are each the size of #unique words
    # from all instances of the respective dataset.
    # List dimensions have: 0: name, 1: ingredients, 2: steps.
    if "countvec" in modes:
        directory = "COMP30027_2021_Project2_datasets/recipe_text_features_countvec/"
        file_names = ("train_name_countvectorizer.pkl", "train_ingr_countvectorizer.pkl", "train_steps_countvectorizer.pkl")
        count_dicts = [pickle.load(open(directory + file_name, "rb")).vocabulary_ for file_name in file_names]

        test_file_names = ("test_name_vec.npz", "test_ingr_vec.npz", "test_steps_vec.npz")
        train_file_names = ("train_name_vec.npz", "train_ingr_vec.npz", "train_steps_vec.npz")

        test_BoW_matrices = [scipy.sparse.load_npz(directory + file_name) for file_name in test_file_names]
        train_BoW_matrices = [scipy.sparse.load_npz(directory + file_name) for file_name in train_file_names]
        data["train"]["countvec"] = count_dicts
        data["test"]["BoW"] = test_BoW_matrices
        data["train"]["BoW"] = train_BoW_matrices
    
    # data[set[doc50]]: List of pandas dataframes each with a 50 dimensional vector per instance. 
    # List dimensions have: 0: name, 1: ingredients, 2: steps.
    if "doc50" in modes:
        directory = "COMP30027_2021_Project2_datasets/recipe_text_features_doc2vec50/"
        test_file_names = ("test_name_doc2vec50.csv", "test_ingr_doc2vec50.csv", "test_steps_doc2vec50.csv")
        train_file_names = ("train_name_doc2vec50.csv", "train_ingr_doc2vec50.csv", "train_steps_doc2vec50.csv")
        test_doc50 = [pd.read_csv(directory + file_name, index_col = False, delimiter = ',', header=None) for file_name in test_file_names]
        train_doc50 = [pd.read_csv(directory + file_name, index_col = False, delimiter = ',', header=None) for file_name in train_file_names]
        data["test"]["doc50"] = test_doc50
        data["train"]["doc50"] = train_doc50
    
    # data[set[doc100]]: List of pandas dataframes each with a 100 dimensional vector per instance. 
    # List dimensions have: 0: name, 1: ingredients, 2: steps.
    if "doc100" in modes:
        directory = "COMP30027_2021_Project2_datasets/recipe_text_features_doc2vec100/"
        test_file_names = ("test_name_doc2vec100.csv", "test_ingr_doc2vec100.csv", "test_steps_doc2vec100.csv")
        train_file_names = ("train_name_doc2vec100.csv", "train_ingr_doc2vec100.csv", "train_steps_doc2vec100.csv")
        test_doc100 = [pd.read_csv(directory + file_name, index_col = False, delimiter = ',', header=None) for file_name in test_file_names]
        train_doc100 = [pd.read_csv(directory + file_name, index_col = False, delimiter = ',', header=None) for file_name in train_file_names]
        data["test"]["doc100"] = test_doc100
        data["train"]["doc100"] = train_doc100
    return data

    # When all imports are active data should have the structure: {"Test": {...}, "Train": {...}}
        # data["Test"] = {"n_steps": (pd.series), "n_ingr": (pd.series), "BoW": (some DF type Bag of Words vectors), "doc50": (pd.DataFrame 50 dimensions),
        #   "doc100": (pd.DataFrame 100 dimensions)}
        # data["Train"] = {"duration": (pd.series categorical), "n_steps": (pd.series), "n_ingr": (pd.series), "countvec": (dict of total word counts across 
        #   all instances), "BoW": (some DF type Bag of Words vectors), "doc50": (pd.DataFrame 50 dimensions), "doc100": (pd.DataFrame 100 dimensions)}