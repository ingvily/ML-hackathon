

#http://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/


# Multilayer Perceptron to Predict International Airline Passengers (t+1, given t)
import matplotlib.pyplot as plt
import pandas
import math
import csv
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


batch_size = 1


def generateDataset(): 
	print "Generating dataset...."
	X = []
	y = []

	train_file = "../traindata/train_scaled.csv"

	train = np.loadtxt( train_file, delimiter = ',' )

	X = train[:,0:-1]
	y = train[:,-1]
	y = y.reshape( -1, 1 )

	trainSize = len(X)/4*3;

	X_train = X[:trainSize]
	y_train = y[:trainSize]
	X_val = X[trainSize:]
	y_val = y[trainSize:]

	# reshape input to be [samples, time steps, features]
	X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
	X_val = np.reshape(X_val, (X_val.shape[0], 1, X_val.shape[1]))
	return (X_train, X_val, y_train, y_val)


def createModelExample1(X_train, y_train):
	print "Creating and training model..."
	inputSize = X_train.shape[2];
	# create and fit the LSTM network
	model = Sequential()
	model.add(LSTM(4, input_dim=inputSize))
	#model.add(Dense(50, input_dim=26))
	#model.add(Dense(25))
	model.add(Dense(1))
	adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	model.compile(loss='mean_squared_error', optimizer=adam)
	hist = model.fit(X_train, y_train, nb_epoch=5, batch_size=1, verbose=2, shuffle=False, validation_split=0.2 )
	print(hist.history)
	return model

def createModelExample2(X_train, y_train):
	inputSize = X_train.shape[2];
	model = Sequential()
	#model.add(LSTM(4, batch_input_shape=(batch_size, 1 , inputSize), stateful=True, return_sequences=True))
	model.add(LSTM(32, batch_input_shape=(batch_size, 1 , inputSize), stateful=True))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	for i in range(1):
		model.fit(X_train, y_train, nb_epoch=1, batch_size=batch_size, verbose=2, shuffle=False)
		model.reset_states()
	return model

def estimatePerformanceExample1(model, X_train, X_val, y_train, y_val):
	print "Estimate performance..."
	# Estimate model performance
	trainScore = model.evaluate(X_train, y_train, verbose=0)
	print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
	testScore = model.evaluate(X_val, y_val, verbose=0)
	print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))

def estimatePerformanceExample2(model, X_train, X_val, y_train, y_val):
	trainPredict = model.predict(X_train, batch_size=batch_size)
	model.reset_states()
	testPredict = model.predict(X_val, batch_size=batch_size)
	# calculate root mean squared error
	trainScore = math.sqrt(mean_squared_error(y_train, trainPredict[:,0]))
	print('Train Score: %.2f RMSE' % (trainScore))
	testScore = math.sqrt(mean_squared_error(y_val, testPredict[:,0]))
	print('Test Score: %.2f RMSE' % (testScore))



X_train, X_val, y_train, y_val = generateDataset()
model = createModelExample(X_train, y_train)
estimatePerformanceExample(model, X_train, X_val, y_train, y_val)





