from sklearn.metrics import mean_squared_error as MSE
import csv
import numpy as np
from math import sqrt

y = []
y_predicted = []

def readDataSet():
    with open('../testdata/test_scaled.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
                y.append(float(row[-1]))
                
def readCSV():
    with open('predicition.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
                y_predicted.append(row[-1]) 
                

readDataSet()

for i in range(29):
    y_predicted.append(0.1);
    
mse = MSE( y, y_predicted )
rmse = sqrt( mse )

print rmse