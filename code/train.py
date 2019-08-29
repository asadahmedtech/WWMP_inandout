import pandas as pd 
import numpy as np 
import os
import matplotlib.pyplot as plt

from dataloader import load_data, train_val_test_split
from models import create_network
from utils import *

from keras.callbacks import TensorBoard
from keras import losses
from keras import optimizers

import argparse

parser = argparse.ArgumentParser(
    description='Train Waste Water Treatment Plant',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--data_file', default='data.csv', type=input_dir, help='Data file for training.')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate') # NOTE change for diff models
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--split_ratio', default=0.3, type=float, help='split ratio for traing and validation')
parser.add_argument('--resume', '-r', type=int, default=0, help='resume from checkpoint')
parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--momentum', '-lm', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-ld', type=float, default=0.001, help='Weight decay (L2 penalty).')
parser.add_argument('--verbose', type=bool, default = True, help='Print the debug arguments.')
args = parser.parse_args()

_start_time = get_start_time()

hyper_param_search = [{'input_dims' : 10, 'output_dims' : 1, 'neuron_layer' : [50, 20, 10], 'dropout_threshold' : [0, 0.5, 0.5, 0.5], 'regularizer_threshold' : 0.01, 'opt' : optimizers.SGD(lr=args.lr, clipnorm=1.)},
	{'input_dims' : 10, 'output_dims' : 1, 'neuron_layer' : [1000, 500, 100, 50, 10], 'dropout_threshold' : [0, 0.3, 0.5, 0, 0.4, 0.5], 'regularizer_threshold' : 0.01, 'opt' : optimizers.Adam(lr=args.lr)},
	{'input_dims' : 10, 'output_dims' : 1, 'neuron_layer' : [500, 200, 100, 10], 'dropout_threshold' : [0, 0.50, 0.5, 0.6, 0.5, 0.4], 'regularizer_threshold' : 0.01, 'opt' : optimizers.SGD(lr=args.lr, clipnorm=1.)},
	{'input_dims' : 10, 'output_dims' : 3, 'neuron_layer' : [50, 20, 10], 'dropout_threshold' : [0, 0.5, 0.5, 0.5], 'regularizer_threshold' : 0.01, 'opt' : optimizers.SGD(lr=args.lr, clipnorm=1.)},
	{'input_dims' : 10, 'output_dims' : 3, 'neuron_layer' : [1000, 500, 100, 50, 10], 'dropout_threshold' : [0, 0.3, 0.5, 0, 0.4, 0.5], 'regularizer_threshold' : 0.01, 'opt' : optimizers.Adam(lr=args.lr)},
	{'input_dims' : 10, 'output_dims' : 14, 'neuron_layer' : [100, 50, 5, 50, 100], 'dropout_threshold' : [0, 0.3, 0.3, 0.3, 0.3, 0.3], 'regularizer_threshold' : 0.01, 'opt' : optimizers.SGD(lr=args.lr, clipnorm=1.)},
	{'input_dims' : 10, 'output_dims' : 14, 'neuron_layer' : [200, 100, 50, 100, 200], 'dropout_threshold' : [0, 0.3, 0.3, 0.3, 0.3, 0.3], 'regularizer_threshold' : 0.01, 'opt' : optimizers.Adam(lr=args.lr)},
	]

import tensorflow as tf
from keras import backend as k
 
###################################
# TensorFlow wizardry
config = tf.ConfigProto()
 
# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
 
# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.3
 
# Create a session with the above options specified.
k.tensorflow_backend.set_session(tf.Session(config=config))

# Running Multiple Models for Training and saving the files
for model_ver in hyper_param_search:
	# Loading the Data for training and spliting it
	train_data, target_data  = load_data(args.data_file, model_ver['input_dims'], model_ver['output_dims'], verbose = args.verbose)
	X_train, Y_train, X_val, Y_val, X_test, Y_test = train_val_test_split(train_data, target_data, split_ratio = args.split_ratio, verbose = args.verbose)

	# The tensorboard log directory will be a unique subdirectory based on the start time fo the run
	TBLOGDIR="../logs/{}".format(get_start_time())

	# Defining the callbacks for training 
	tensorboard = TensorBoard(log_dir=TBLOGDIR, histogram_freq=0, batch_size=args.batch_size)

	model = create_network(model_ver['input_dims'], model_ver['output_dims'], model_ver['neuron_layer'], model_ver['dropout_threshold'], model_ver['regularizer_threshold'])

	model.compile(loss = losses.mean_squared_error, optimizer = model_ver['opt'])
	model.fit(X_train, Y_train, epochs = args.epochs, verbose = args.verbose, validation_data = (X_val, Y_val), callbacks = [tensorboard])
	scores = model.evaluate(X_test, Y_test, verbose=args.verbose)

	print("===> Scores : ", model_ver, scores)
