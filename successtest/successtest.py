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
    with open('example.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
                y_predicted.append(float(row[0])) 

                

readDataSet()
readCSV()
    
mse = MSE( y, y_predicted )
rmse = sqrt( mse )

print rmse