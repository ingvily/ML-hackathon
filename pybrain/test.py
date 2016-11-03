import numpy as np
from math import sqrt
from pybrain.datasets.supervised import SupervisedDataSet as SDS
from sklearn.metrics import mean_squared_error as MSE
from trening import net

test_file = '../testdata/test_scaled.csv'

# load data

test = np.loadtxt( test_file, delimiter = ',' )
x_test = test[:,0:-1]
y_test = test[:,-1]
y_test = y_test.reshape( -1, 1 )

# you'll need labels. In case you don't have them...
y_test_dummy = np.zeros( y_test.shape )

input_size = x_test.shape[1]
target_size = y_test.shape[1]

assert( net.indim == input_size )
assert( net.outdim == target_size )

# prepare dataset

ds = SDS( input_size, target_size )
ds.setField( 'input', x_test )
ds.setField( 'target', y_test_dummy )

# predict

p = net.activateOnDataset( ds )

mse = MSE( y_test, p )
rmse = sqrt( mse )

for s in y_test[:10]:
    print (s +1)/2 *(784.83 - 35.02) + 35.02

for s in p[:10]:
    print (s +1)/2 *(784.83 - 35.02) + 35.02
print "testing RMSE:", rmse

