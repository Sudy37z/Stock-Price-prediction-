#importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing dataset

dataset_train = pd.read_csv("D:\stock predictions using rnn\L&T_train.csv")

#Creating an array as RNN works only on array

training_set = dataset_train.iloc[:,1:2].values

# feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range= (0,1))

training_set_scaled = sc.fit_transform(training_set)

#create a data structure with 60 timesteps and 1 output
X_train = []
y_train = []

for i in range(60,308):
    X_train.append(training_set_scaled[i-60:i,0])
    y_train.append(training_set_scaled[i,0])
X_train,y_train = np.array(X_train),np.array(y_train)

# Reshaping for better indication

X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))

# Importing the keras libraries 
from keras.models import Sequential
from keras. layers import Dense
from keras. layers import LSTM
from keras. layers import Dropout

regressor = Sequential()
# Adding the first LSTM layer and some Dropout regularisation

regressor.add(LSTM(units = 50, return_sequences=True,input_shape=(X_train.shape[1],1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the regressor

regressor.add(Dense(units = 1))

#Compiling the layers

regressor.compile(optimizer='adam', loss = 'mean_squared_error')

#Fitting the model

regressor.fit(X_train,y_train, epochs = 100, batch_size=32)


#Real stock price of 2017

dataset_test = pd.read_csv("D:\stock predictions using rnn\L&T_test.csv")
real_stock_price = dataset_test.iloc[:,1:2].values






