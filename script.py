import scipy
import pandas as pd
import pickle

def import_data():
    train = pd.read_csv("COMP30027_2021_Project2_datasets/recipe_train.csv")
    
    # Import bag of words data
    directory = "COMP30027_2021_Project2_datasets/recipe_text_features_countvec/"
    file_names = ("/train_name_countvectorizer.pkl", "/train_ingr_countvectorizer.pkl", "/train_steps_countvectorizer.pkl")
    dicts = [pickle.load(open(directory + file_name, "rb")).vocabulary_ for file_name in file_names]

    file_names = ("/train_name_vec.npz", "/train_ingr_vec.npz", "/train_steps_vec.npz")
    BoWmatrices = [scipy.sparse.load_npz(directory + file_name) for file_name in file_names]

    # Import doc2vec 50 data
    directory = "COMP30027_2021_Project2_datasets/recipe_text_features_doc2vec50/"
    file_names = ("train_name_doc2vec50.csv", "train_ingr_doc2vec50.csv", "train_steps_doc2vec50.csv")
    doc50 = [pd.read_csv(file_name, index_col = False, delimiter = ',', header=None) for file_name in file_names]
    
    # Import doc2vec 100 data
    directory = "COMP30027_2021_Project2_datasets/recipe_text_features_doc2vec100"
    file_names = ("train_name_doc2vec50.csv", "train_ingr_doc2vec50.csv", "train_steps_doc2vec50.csv")
    doc100 = [pd.read_csv(file_name, index_col = False, delimiter = ',', header=None) for file_name in file_names]
        
    return dicts, BoWmatrices, doc50, doc100
