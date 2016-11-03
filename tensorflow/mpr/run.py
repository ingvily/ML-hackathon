import tensorflow as tf
from tensorflow.contrib import learn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn import datasets, linear_model
from sklearn import cross_validation
import numpy as np
from sklearn.metrics import mean_squared_error as MSE
from math import sqrt

data_train = pd.read_csv('./train_scaled.csv', header=None, sep=',').as_matrix()
data_test = pd.read_csv('./test_scaled.csv', header=None, sep=',').as_matrix() #TODO!!!!!!! 
###INGVILD AWS!!!! 

length = len(data_train[0])

X_train = data_train[:, 0 : length-1]
Y_train = data_train[:, length-1 : length]

X_test = data_test[:, 0 : length-1]
Y_test = data_test[:, length-1 : length] 

total_len = X_train.shape[0]

# Parameters
learning_rate = 0.0001
training_epochs = 5
batch_size = 10
display_step = 1

n_hidden_1 = 10 # 1st layer number of features
n_hidden_2 = 51 # 2nd layer number of features

n_input = X_train.shape[1]
n_classes = 1

# tf Graph input
x = tf.placeholder("float", [None, 26])
y = tf.placeholder("float", [None, 1])

# Create model

def accuracy(predicted, fasit):
    return np.sum(np.equal(np.argmax(predicted, 1), np.argmax(fasit, 1)))*1.0 / len(fasit)

def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], 0, 0.1)),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], 0, 0.1)),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes], 0, 0.1))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1], 0, 0.1)),
    'b2': tf.Variable(tf.random_normal([n_hidden_2], 0, 0.1)),
    'out': tf.Variable(tf.random_normal([n_classes], 0, 0.1))
}

# Construct model
pred_temp = multilayer_perceptron(x, weights, biases)
pred = tf.transpose(pred_temp)

# Define loss and optimizer
cost = tf.reduce_mean(tf.square(pred-y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Launch the graph
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(total_len/batch_size)
        # Loop over all batches
        for i in range(total_batch-1):
            batch_x = X_train[i*batch_size:(i+1)*batch_size]
            batch_y = Y_train[i*batch_size:(i+1)*batch_size]

            # Run optimization op (backprop) and cost op (to get loss value)
            _, c, estimate = sess.run([optimizer, cost, pred_temp], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch

        # sample prediction
        label_value = batch_y
        err = label_value-estimate
        print ("num batch:", total_batch)
         # Display logs per epoch step
        if epoch % display_step == 0:
            print ("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))
            print ("[*]----------------------------")
            for i in xrange(3):
                print ("label value:", label_value[i], \
                    "estimated value:", estimate[i][0])
            print ("[*]============================")

    _, _, estimate = sess.run([optimizer, cost, pred_temp], feed_dict={x: X_test,
                                                          y: Y_test})

    mse = MSE( Y_test, estimate )
    rmse = sqrt( mse )
    print (rmse)

