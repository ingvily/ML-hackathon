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

    def __init__(self, training_percentage):
        self._data_and_labels = self.load_data_set('./train_bool.csv', ',').as_matrix()

        self._length_of_input_array = len(self._data_and_labels[0])

        self._all_data = self._data_and_labels[:, 0 : self._length_of_input_array - 2]
        self._all_labels = self._data_and_labels[:, self._length_of_input_array -2 : self._length_of_input_array]

        self.size_of_data_set = len(self._all_data)
        perm = np.arange(self.size_of_data_set)
        np.random.shuffle(perm)
        self._all_data = self._all_data[perm]
        self._all_labels = self._all_labels[perm]

        split_training_index = int(self.size_of_data_set * training_percentage)

        self._training_data = self._all_data[0 : split_training_index]
        self._training_labels = self._all_labels[0 : split_training_index]

        self._testing_data = self._all_data[split_training_index : self.size_of_data_set]
        self._testing_labels = self._all_labels[split_training_index : self.size_of_data_set]
        
        self.numer_of_input_nodes = len(self._all_data[0])
        self.number_of_output_nodes = len(self._training_labels[0]) 

        self._current_index = 0
        self._size_of_training = len(self._training_data)




    

    