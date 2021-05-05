from collections import Counter

def zeroR(train, test):
	counts = Counter(train)
	most_frequent = counts.most_common(1)[0][0]
	return [most_frequent] * len(test)