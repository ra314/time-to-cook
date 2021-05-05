from collections import Counter

from ImportData import import_data

def zeroR_predict(train_labels, n):
	counts = Counter(train_labels)
	most_frequent = counts.most_common(1)[0][0]
	return [most_frequent] * n

def zeroR_evaluate(train_labels, test_labels):
	predictions = zeroR_predict(train_labels, len(test_labels))
	return 100*sum(pred == label for pred, label in zip(predictions, test_labels))/len(test_labels)

def baseline( test_size = 0.8, RS = 1):
	data = import_data(["0-R"], test_size, RS)
	return zeroR_evaluate(data[0], data[1])
