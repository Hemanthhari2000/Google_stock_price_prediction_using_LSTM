# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import warnings 
warnings.filterwarnings('ignore')

# Importing the training set
dataset_train = pd.read_csv('./dataset/Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timestamps and 1 output
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping 
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Building the RNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout 

def create_model():
	#regressor is used because we are predicting a continuous output instead of predicting classes
	regressor = Sequential()

	# adding first LSTM Layer with some reguralization
	regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
	regressor.add(Dropout(0.2))

	# adding second LSTM Layer with some reguralization
	regressor.add(LSTM(units = 50, return_sequences = True))
	regressor.add(Dropout(0.2))

	# adding third LSTM Layer with some reguralization
	regressor.add(LSTM(units = 50, return_sequences = True))
	regressor.add(Dropout(0.2))

	# adding fourth LSTM Layer with some reguralization
	regressor.add(LSTM(units = 50))
	regressor.add(Dropout(0.2))

	# adding the output Layer
	regressor.add(Dense(units = 1))

	# compiling the NN
	regressor.compile(optimizer = 'rmsprop', loss = 'mean_squared_error')

	return regressor

regressor = create_model()

# Fitting the RNN to the Model/Data's
regressor.fit(X_train, y_train, epochs = 100, batch_size = 64, verbose=1)

# Save the Model
# regressor.save('regressor_model.h5')

# Making a Prediction and visualising
# Importing the training set
dataset_test = pd.read_csv('./dataset/Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'] , dataset_test['Open']), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)

X_test = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

predicted_stock_price = np.array(predicted_stock_price)

print(f'Predictions:\n{predicted_stock_price}')

























