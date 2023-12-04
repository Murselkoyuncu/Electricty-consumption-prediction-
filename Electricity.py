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
history = Model.fit(x_train, y_train, epochs= 25 , batch_size=32, validation_split=0.2)

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

prediction_test = []
Batch_one = training_set_scaled[-24:]
New_batch = Batch_one.reshape(1,24,3)

for i in range (48):
    First_pred = Model.predict(New_batch)[0]
    prediction_test.append(First_pred)
    New_var = test_set_scaled[i,:]
    New_var = New_var.reshape(1,2)
    New_test = np.insert(New_var,2,[First_pred],axis=1)
    New_test = New_test.reshape(1,1,3)
    New_batch= np.append(New_batch[:,1:,:],New_test,axis=1)
    
prediction_test = np.array(prediction_test)

SI = MinMaxScaler(feature_range=(0,1))
y_scale = training_set[:,2:3]
SI.fit_transform(y_scale)
predictions = SI.inverse_transform(prediction_test)
real_values = test_set[:,2]
plt.plot(real_values,color='red',label = 'actaul value of electrical consumption')
plt.plot(predictions,color='blue',label = 'predicted Values')
plt.plot('electrical consumption prediction')
plt.xlabel('time(h)')
plt.ylabel('electrical demand(MW)')
plt.legend()
plt.show()

import math 
from sklearn.metrics import mean_squared_error

RMSE = math.sqrt(mean_squared_error(real_values,predictions))



def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


MAPE = mean_absolute_percentage_error(real_values, predictions) 
# Load the model
Model = load_model('LSTM-Multivariate') 


