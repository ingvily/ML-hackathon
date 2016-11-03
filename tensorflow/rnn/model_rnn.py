import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.contrib import learn
import pandas as pd
import numpy as np

def split_data(data, val_size=0.1, test_size=0.1):
	ntest = int(round(len(data) * (1 - test_size)))
	nval = int(round(len(data.iloc[:ntest]) * (1 - val_size)))
	df_train, df_val, df_test = data.iloc[:nval], data.iloc[nval:ntest], data.iloc[ntest:]
	return df_train, df_val, df_test

def rnn_data(data, time_steps, labels=False):
	rnn_df = []
	for i in range(len(data) - time_steps):
		data_ = data.iloc[i: i + time_steps].as_matrix()
		rnn_df.append(data_ if len(data_.shape) > 1 else [[i] for i in data_])
	return np.array(rnn_df)


def prepare_data(data, time_steps, labels=False, val_size=0.05, test_size=0.05):
	df_train, df_val, df_test = split_data(data, val_size, test_size)
	return (rnn_data(df_train, time_steps, labels=labels), rnn_data(df_val, time_steps, labels=labels), rnn_data(df_test, time_steps, labels=labels))


def load_csvdata(data, time_steps, seperate=False):
	if not isinstance(data, pd.DataFrame):
		data = pd.DataFrame(data)
	train_x, val_x, test_x = prepare_data(data['a'] if seperate else data, time_steps)
	train_y, val_y, test_y = prepare_data(data['b'] if seperate else data, time_steps, labels=True)
	return dict(train=train_x, val=val_x, test=test_x), dict(train=train_y, val=val_y, test=test_y)


def lstm_model(time_steps, rnn_layers, dense_layers=None):
	def lstm_cells(layers):
		if isinstance(layers[0], dict):
			return [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(layer['steps'], state_is_tuple=True), layer['keep_prob'])
				if layer.get('keep_prob') else tf.nn.rnn_cell.BasicLSTMCell(layer['steps'], state_is_tuple=True)
				for layer in layers]
		return [tf.nn.rnn_cell.BasicLSTMCell(steps, state_is_tuple=True) for steps in layers]


	def dnn_layers(input_layers, layers):
		if layers and isinstance(layers, dict):
			return learn.ops.dnn(input_layers, layers['layers'], activation=layers.get('activation'),dropout=layers.get('dropout'))
		elif layers:
			return learn.ops.dnn(input_layers, layers)
		else:
			return input_layers


	def _lstm_model(X, y):
		stacked_lstm = tf.nn.rnn_cell.MultiRNNCell(lstm_cells(rnn_layers), state_is_tuple=True)
		x_ = learn.ops.split_squeeze(1, time_steps, X)
		output, layers = tf.nn.rnn(stacked_lstm, x_, dtype=dtypes.float32)
		output = dnn_layers(output[-1], dense_layers)
		return learn.models.linear_regression(output, y)
