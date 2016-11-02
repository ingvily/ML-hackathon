from trening import mlp

X_test = []
y_test = []

def readDataSet():
    with open('../testdata/test_scaled.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
             #if(isinstance(row[0], float)):
                row = list(map(lambda x: float(x), row))
                X_test.append(row[:-1])
                y_test.append(row[-1])
                
                
readDataSet()


p = mlp.predict(X_test);
mse = MSE( y_test, p )
rmse = sqrt( mse )
print rmse

#print("Test set score: %f" % mlp.score(X_test, y_test))