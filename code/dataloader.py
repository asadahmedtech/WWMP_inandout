import pandas as pd 
import numpy as np 
import os

from sklearn import preprocessing 
from sklear.model_selection import train_test_split


def load_data(filename, input_dims, output_dims, filepath = '../dataset/', normalize = False, verbose = False):

	# Loadomg the CSV files 
	if(verbose):
		print("==> Loading CSV dataset")
	data = pd.read_csv(os.path.join(filepath, filename))

	#Target labels
	target_labels = data.columns[input_dims:input_dims + output_dims]
	other_labels = data.columns[input_dims + output_dims:]

	# Train and Test Data
	target_data = data[target_labels]
	train_data = data.drop(target_labels, axis = 1)

	if(other_labels != []):
		train_data = data.drop(other_labels, axis = 1)

	del data

	if(normalize):
		if(verbose):
			print("==> Normalizing CSV dataset")
		min_max_scaler = preprocessing.MinMaxScaler()
		train_data = min_max_scaler.fit_transform(train_data)

	return train_data, target_data

def train_val_test_split(train_data, target_data, split_ratio = 0.3, verbose = False):

	# Train, Valid, Test split
	if(verbose):
		print("==> Spliting into Train, Valid, Test")
	X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(train_data, target_data, test_size = split_ratio)
	X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size = 0.5)

	del train_data, target_data, X_val_and_test, Y_val_and_test

	if(verbose):
		print("==> Shape Sizes \n X_train : {} \n Y_train : {} \n X_validation : {} \n Y_validation : {} \n \
			X_test : {} \n Y_test : {} \n".format(X_train.shape, Y_train.shape, X_val.shape, Y_val.shape, X_test.shape, Y_test.shape))

	return X_train, Y_train, X_val, Y_val, X_test, Y_test






