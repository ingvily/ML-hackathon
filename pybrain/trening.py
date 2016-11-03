from pybrain.datasets.supervised import SupervisedDataSet as SDS
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import SigmoidLayer, LinearLayer
from pybrain.tools.shortcuts import buildNetwork
import csv
import numpy as np
from math import sqrt



def prepareDataset():


	train_file = "../traindata/train_scaled.csv"
	train = np.loadtxt( train_file, delimiter = ',' )

	x_train = train[:,0:-1]
	y_train = train[:,-1]
	y_train = y_train.reshape( -1, 1 )

	input_size = x_train.shape[1]
	target_size = y_train.shape[1]

	print input_size
	print target_size

	# prepare dataset

	ds = SDS( input_size, target_size )
	ds.setField( 'input', x_train )
	ds.setField( 'target', y_train )
	return (ds, input_size)

def createModel(ds, input_size):
	#----------
	# build the network
	#----------
	net = buildNetwork(input_size,
	                   100, # number of hidden units
	                   1,
	                   bias = True,
	                   hiddenclass = SigmoidLayer,
	                   outclass = LinearLayer
	                   )
	#----------
	# train
	#----------

	hidden_size = 100
	epochs = 10
	continue_epochs = 10
	validation_proportion = 0.25


	trainer = BackpropTrainer(net, ds, verbose = True)
	#trainer.trainUntilConvergence(maxEpochs = 500)
	train_mse, validation_mse = trainer.trainUntilConvergence( verbose = True, validationProportion = validation_proportion, 
	                                                          maxEpochs = epochs, continueEpochs = continue_epochs )
	return net
	#epochs = 500
	#print "training for {} epochs...".format( epochs )
	#for i in range( epochs ):
	#	mse = trainer.train()
	#	rmse = sqrt( mse )
	#	print "training RMSE, epoch {}: {}".format( i + 1, rmse )

ds, input_size = prepareDataset()
net = createModel(ds, input_size);