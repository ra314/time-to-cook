import pandas as pd
import numpy as np

def predict_for_kaggle(model, test_data):
	predictions = model.predict(test_data)
	df = pd.DataFrame({"id": np.arange(1, len(predictions)+1), "duration_label": predictions})
	df.to_csv("predictions.csv", index = False)
