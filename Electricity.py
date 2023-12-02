# -*- coding: utf-8 -*-
"""
Created on Sat Dec 2 17:24:36 2023

@author: mrslk
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("Electricity.csv")

df.dropna(inplace=True)

training_set = df.iloc[:8712, 1:4].values
test_set = df.iloc[8712:, 1:4].values

# Data scaling
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)
test_set_scaled = sc.transform(test_set)

test_set_scaled = test_set_scaled[:, 0:2]

x_train = []
y_train = []
ws = 24
for i in range(ws, len(training_set_scaled)):
    x_train.append(training_set_scaled[i - ws:i, 0:3])
    y_train.append(training_set_scaled[i, 2])

x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 3))

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

Model = Sequential()

Model.add(LSTM(units=70, return_sequences=True, input_shape=(x_train.shape[1], 3)))
Model.add(Dropout(0.2))

Model.add(LSTM(units=70, return_sequences=True))
Model.add(Dropout(0.2))

Model.add(LSTM(units=70, return_sequences=True))
Model.add(Dropout(0.2))

Model.add(LSTM(units=70))
Model.add(Dropout(0.2))

Model.add(Dense(units=1))

Model.compile(optimizer='adam', loss='mean_squared_error')

# Train your model and collect history
history = Model.fit(x_train, y_train, epochs= 20 , batch_size=32, validation_split=0.2)

# Plot the training loss
plt.plot(range(len(history.history['loss'])), history.history['loss'])
plt.xlabel('epoch number')
plt.ylabel('loss')
plt.show()

# Save the model
Model.save('LSTM-Multivariate')

from keras.models import load_model
# Load the model
Model = load_model('LSTM-Multivariate') 


