from dataset import Dataset
from model import FFNN
import tensorflow as tf
import numpy as np


def accuracy(predicted, fasit):
	return np.sum(np.equal(np.argmax(predicted, 1), np.argmax(fasit, 1)))*1.0 / len(fasit)


def run(): 
	dataset = Dataset(0.7)
	nodes = [dataset.numer_of_input_nodes, 10, 10, dataset.number_of_output_nodes]
	model = FFNN(nodes)
	model.build()
	
	batch_size = 200
	
	sess = tf.Session()
	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
		i = 0
		while i < 100:
			i += 1
			data_training, data_target = dataset.next_training_batch(batch_size)
			_, loss_train = sess.run([model.train, model.xent_mean], feed_dict={model.x: data_training, model.ideal: data_target})
			
		predicted = sess.run(model.y, feed_dict={model.x: dataset._testing_data})
		print 'Testing accuracy', accuracy(predicted, dataset._testing_labels)           


run()