
from sklearn.neural_network import MLPRegressor
import csv
import numpy as np
from sklearn.preprocessing import StandardScaler  
from sklearn.metrics import mean_squared_error as MSE
from math import sqrt

X = []
y = []

def readDataSet():
    with open('../traindata/train_scaled.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
             #if(isinstance(row[0], float)):
                row = list(map(lambda x: float(x), row))
                X.append(row[:-1])
                y.append(row[-1])
                
                
readDataSet()

#scaler = StandardScaler()  
#scaler.fit(X_train)  
#X_train = scaler.transform(X_train)  
#X_test = scaler.transform(X_test)

X_train = X[:1800]
y_train = y[:1800]
X_val = X[1800:]
y_val = y[1800:]



print len(X_train)
print len(X_val)



mlp = MLPRegressor(hidden_layer_sizes=(100,50,), random_state=1, max_iter=1,
                   warm_start=True, learning_rate_init =0.000001)

for i in range(5000):
    mlp.fit(X_train, y_train)
    if(i % 100 == 0):
        #print("Validation set score: %f" % mlp.score(X_val, y_val)) 
        #print("Training set score: %f" % mlp.score(X_train, y_train))
        p = mlp.predict(X_val);
        mse = MSE( y_val, p )
        rmse = sqrt( mse )
        print rmse
    
print("Training set score: %f" % mlp.score(X_train, y_train))


