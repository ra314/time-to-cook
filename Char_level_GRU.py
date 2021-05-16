from ImportData import import_data
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model, Sequential
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, SpatialDropout1D, InputLayer
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
CUDA_VISIBLE_DEVICES=""

from ULMFiT import format_to_text

train, test = import_data(["raw"])
df = pd.concat([train, test], axis=0)
df = format_to_text(df)
X, Y = df['text'], df['duration_label']
Y = pd.get_dummies(df['duration_label']).values

max_len = max([len(item) for item in df['text']])
chars = sorted(set("".join(list(df['text']))))

'''
#Converting the characters into one hot encodes
if (os.path.exists('Encode_to_char_matrix.csv')):
	pd.df = pd.read_csv('Encode_to_char_matrix.csv')
else:
	def encode_to_char_matrix(text):
		matrix = np.zeros((len(text), len(chars)), dtype=np.bool)
		for i in range(len(text)):
			matrix[i][chars.index(text[i])] = 1
		return matrix
		
	df['text'] = df['text'].apply(encode_to_char_matrix)
'''

#le = LabelEncoder()
#Y = le.fit_transform(Y)
#Y = Y.reshape(-1,1)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.15)

#Y_train = np.asarray(Y_train).astype('float32').reshape((-1,1))
#Y_test = np.asarray(Y_test).astype('float32').reshape((-1,1))

tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(X_train)
sequences = tokenizer.texts_to_sequences(X_train)
MAX_SEQUENCE_LENGTH = max([len(item) for item in sequences])
sequences_matrix = sequence.pad_sequences(sequences,maxlen=MAX_SEQUENCE_LENGTH)


model = Sequential()
model.add(Embedding(len(tokenizer.index_word)+1, 100, input_length = MAX_SEQUENCE_LENGTH))
model.add(SpatialDropout1D(0.2))
model.add(GRU(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
model.fit(sequences_matrix,Y_train,batch_size=256,epochs=30, validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss', patience = 3, min_delta=0.0001)])



test_sequences = tokenizer.texts_to_sequences(X_test) 
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=MAX_SEQUENCE_LENGTH)

accr = model.evaluate(test_sequences_matrix,Y_test)

print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
