import csv
import numpy as np
import pandas as pd


class Dataset():
    def next_training_batch(self, batch_size):
        start = self._current_index
        self._current_index += batch_size

        if self._current_index > self._size_of_training:
            perm = np.arange(self._size_of_training)
            np.random.shuffle(perm)
            self._training_data = self._training_data[perm]
            self._training_labels = self._training_labels[perm]

            start = 0
            self._current_index = batch_size
        end = self._current_index
        return self._training_data[start:end], self._training_labels[start:end]



    def load_data_set(self, file, seperator):
        return pd.read_csv(file, header=None, sep=seperator)

    def __init__(self, test_procentage):
        self._scaled_training_data_and_labels = self.load_data_set('./train_scaled_binary.csv', ',').as_matrix()
        self._scaled_test_data_and_labels = self.load_data_set('./test_scaled_binary.csv', ',').as_matrix()
        
        self.length_data = len(self._scaled_training_data_and_labels[0])
        
        self._training_data = self._scaled_training_data_and_labels[:, 0 : self.length_data-2]
        self._training_labels = self._scaled_training_data_and_labels[:, self.length_data-2 : self.length_data]
        
        self._testing_data = self._scaled_test_data_and_labels[:, 0: self.length_data-2]
        self._testing_labels = self._scaled_test_data_and_labels[:, self.length_data-2 : self.length_data]
        
        self.numer_of_input_nodes = len(self._training_data[0]) 
        self.number_of_output_nodes = len(self._training_labels[0]) 
        
        self._current_index = 0
        self._size_of_training = len(self._training_data)




    

    