from ImportData import import_data
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model, Sequential
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, SpatialDropout1D, GRU, Bidirectional
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping

#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from ULMFiT import format_to_text

train, test = import_data(["raw"])
df = pd.concat([train, test], axis=0)
df = format_to_text(df)
X, Y = df['text'], df['duration_label']
Y = pd.get_dummies(df['duration_label']).values

#le = LabelEncoder()
#Y = le.fit_transform(Y)
#Y = Y.reshape(-1,1)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.15)

#Y_train = np.asarray(Y_train).astype('float32').reshape((-1,1))
#Y_test = np.asarray(Y_test).astype('float32').reshape((-1,1))


#max_words = int(len(set(("".join(list(df['text'])).split()))) * 0.1)
#max_len = 300
max_words = 1000
max_len = 150
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train)
sequences = tok.texts_to_sequences(X_train)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

'''
def RNN():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,50,input_length=max_len)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(3,name='out_layer')(layer)
    layer = Activation('softmax')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model
'''


model = Sequential()
model.add(Embedding(max_words, 50, input_length=max_len))
model.add(SpatialDropout1D(0.2))
model.add(Bidirectional(GRU(1024, dropout=0.2, recurrent_dropout=0.2)))
#model.add(Bidirectional(GRU(200, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
#model.add(Bidirectional(GRU(200, dropout=0.2, recurrent_dropout=0.2)))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


#history = model.fit(sequences_matrix, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])


#model = RNN()
model.summary()

model.fit(sequences_matrix,Y_train,batch_size=16,epochs=100, validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss', patience = 10, min_delta=0.0001)])
          
test_sequences = tok.texts_to_sequences(X_test) 
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)

accr = model.evaluate(test_sequences_matrix,Y_test)

print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

# LSTM 100 neurons softmax no rec dropout 44.1% acc on 0.15 test size 0.0001 callback
# LSTM 500 neurons softmax no rec dropout 66.9% acc on 0.15 test size 0.0001 callback
# LSTM 10 neurons softmax no rec dropout 72.4% acc on 0.15 test size 0.0001 callback
# LSTM 5 neurons softmax no rec dropout 50.0% acc on 0.15 test size 0.0001 callback
# GRU 50 neurons softmax no rec dropout 67.6% acc on 0.15 test size 0.0001 callback
# GRU 50 neurons softmax no rec dropout 72.5% acc on 0.15 test size 0.0005 callback
# GRU 50 neurons softmax no rec dropout 71.1% acc on 0.15 test size 0.01 callback
# LSTM 100 neurons softmax rec dropout = 0.1 51.0% acc on 0.15 test size 0.0001 callback
# LSTM 100 neurons softmax rec dropout = 0 76.0% acc on 0.15 test size 0.0001 callback patience = 3
# LSTM 200 neurons softmax rec dropout = 0 74.7% acc on 0.15 test size 0.0001 callback patience = 3
# LSTM 200 neurons softmax rec dropout = 0 78.0% acc on 0.15 test size 0.0001 callback patience = 3 15 epochs
# LSTM 200 + 200 neurons softmax rec dropout = 0 76.5% acc on 0.15 test size 0.0001 callback patience = 3 10 epochs
#   Adding another 10 epochs: 78.0%, did not finish early.
#   Adding another 10 epochs: 78.4%. Ended early.
# LSTM 200 + 200 + 200 neurons softmax rec dropout = 0 78.8% acc on 0.15 test size 0.0001 callback patience = 3 10 epochs did not finish early.
#   Adding another 20 epochs: Stopped after 6. 77.9%
#   Adding another 20 epochs: Stopped after 10. 79.8% 
# GRU 200 neurons softmax rec dropout = 0 78.1% acc on 0.15 test size 0.0001 callback patience = 5 20 epochs did not finish early.
#    Adding another 20 epochs: Stopped after 12. 79.2%
# GRU 200 + 200 neurons softmax rec dropout = 0 79.8% acc on 0.15 test size 0.0001 callback patience = 5 20 epochs finished at 17.
#   Adding another 20 epochs: Stopped after 14. 79.8%.
#   Adding another 20 epochs: Stopped after 6: 79.4%
# BID LSTM 200 neurons softmax rec dropout = 0 78.0% acc on 0.15 test size 0.0001 callback patience = 5 20 epochs did not finish early.
# BID GRU 200 neurons softmax rec dropout = 0 75.9% acc on 0.15 test size 0.0001 callback patience = 5 20 epochs finished at 19. Promising around 15 epochs.
# BID LSTM 200 + 200 neurons softmax rec dropout = 0 79.8% acc on 0.15 test size 0.0001 callback patience = 5 30 epochs finished at 24.
# BID GRU 200 + 200 neurons softmax rec dropout = 0 80.3% acc on 0.15 test size 0.0001 callback patience = 5 30 epochs finished at 13. Not sure why it stopped though.
#   Another 30 epochs: Stopped after 7. 79.0%.
# Model1: BID GRU 200 + 200 neurons softmax rec dropout = 0.2 on both layers 0.80% acc on 0.15 test size 0.0001 callback 
#	Did't continue ffurther because val acc had been decreasing for a while This was at epoch 13.
# Model2: Used a smaller bidirectional LSTM 128 neurons softmax rec dropout = 0.2
#	Changed the number of words and the max len to something more reasonable
#	Stopped at Epoch 15 Max Val acc was 0.8019
