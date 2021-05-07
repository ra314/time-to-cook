import scipy
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

def import_data(modes = ["all"], test_size = 0.8, RS = 1):
	""" 
	test size: Size ratio of test partition in train_test_split(). RS: Random state integer for train_test_split().
	When all imports are active data should have the structure: {"test": {...}, "train": "train": {...}, "test": {...}}
		data["test"] = {"n_steps": (pd.series), "n_ingr": (pd.series), "BoW": (some DF type Bag of Words vectors), "doc50": (pd.DataFrame 50 dimensions),"doc100": (pd.DataFrame 100 dimensions)}
		data["train"]["train" / "test"] = {"duration": (pd.series categorical), "n_steps": (pd.series), "n_ingr": (pd.series), "countvec": (dict of total word counts across all instances), "BoW": (some DF type Bag of Words vectors), "doc50": (pd.DataFrame 50 dimensions), "doc100": (pd.DataFrame 100 dimensions)}
	"""
	data = {"test": {}, "train": {"train": {}, "test": {}}}
	test = pd.read_csv("COMP30027_2021_Project2_datasets/recipe_test.csv")
	train = pd.read_csv("COMP30027_2021_Project2_datasets/recipe_train.csv")
	X_train, X_test = train_test_split(train, test_size = test_size, random_state = RS)

	# Pandas series
	data["train"]["train"]["duration"] = X_train["duration_label"]
	data["train"]["test"]["duration"] = X_test["duration_label"]
	if "0-R" in modes:
		return [data["train"]["train"]["duration"], data["train"]["test"]["duration"]]

	# Pandas series
	data["train"]["train"]["n_steps"] = X_train["n_steps"]
	data["train"]["test"]["n_steps"] = X_test["n_steps"]
	data["test"]["n_steps"] = test["n_steps"]

	# Pandas series
	data["train"]["train"]["n_ingr"] = X_train["n_ingredients"]
	data["train"]["test"]["n_ingr"] = X_test["n_ingredients"]
	data["test"]["n_ingr"] = test["n_ingredients"]

	# Importing all data
	if "all" in modes:
		modes = ["countvec", "doc50", "doc100"]

	# data[train][train/test][countvec]: List of dictionaries of wordcounts per instance for training set.
	# data[set]([train/test])[BoW]: List of vectors of wordcounts per instance. Vectors are each the size of unique words
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

		tuples = [train_test_split(matrix, test_size = test_size, random_state = RS) for matrix in train_BoW_matrices]
		data["train"]["train"]["BoW"], data["train"]["test"]["BoW"] = zip(*tuples)
		data["test"]["BoW"] = test_BoW_matrices
		data["train"]["train"]["countvec"] = count_dicts

	# data[set]([train/test])[doc50]: List of pandas dataframes each with a 50 dimensional vector per instance.
	# List dimensions have: 0: name, 1: ingredients, 2: steps.
	if "doc50" in modes:
		directory = "COMP30027_2021_Project2_datasets/recipe_text_features_doc2vec50/"
		test_file_names = ("test_name_doc2vec50.csv", "test_ingr_doc2vec50.csv", "test_steps_doc2vec50.csv")
		train_file_names = ("train_name_doc2vec50.csv", "train_ingr_doc2vec50.csv", "train_steps_doc2vec50.csv")
		test_doc50 = [pd.read_csv(directory + file_name, index_col = False, delimiter = ',', header=None) for file_name in test_file_names]
		train_doc50 = [pd.read_csv(directory + file_name, index_col = False, delimiter = ',', header=None) for file_name in train_file_names]

		tuples = [train_test_split(DF, test_size = test_size, random_state = RS) for DF in train_doc50]
		data["train"]["train"]["doc50"], data["train"]["test"]["doc50"] = zip(*tuples)
		data["test"]["doc50"] = test_doc50


	# data[set]([train/test])[doc100]: List of pandas dataframes each with a 100 dimensional vector per instance.
	# List dimensions have: 0: name, 1: ingredients, 2: steps.
	if "doc100" in modes:
		directory = "COMP30027_2021_Project2_datasets/recipe_text_features_doc2vec100/"
		test_file_names = ("test_name_doc2vec100.csv", "test_ingr_doc2vec100.csv", "test_steps_doc2vec100.csv")
		train_file_names = ("train_name_doc2vec100.csv", "train_ingr_doc2vec100.csv", "train_steps_doc2vec100.csv")
		test_doc100 = [pd.read_csv(directory + file_name, index_col = False, delimiter = ',', header=None) for file_name in test_file_names]
		train_doc100 = [pd.read_csv(directory + file_name, index_col = False, delimiter = ',', header=None) for file_name in train_file_names]

		tuples = [train_test_split(DF, test_size = test_size, random_state = RS) for DF in train_doc100]
		data["train"]["train"]["doc100"], data["train"]["test"]["doc100"] = zip(*tuples)
		data["test"]["doc100"] = test_doc100

	return data
