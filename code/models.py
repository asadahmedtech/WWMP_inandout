from keras.model import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers

# Multiple Models 

def create_network(input_dims, output_dims, neuron_layers, dropout_threshold, regularizer_threshold = 0.01):

	layers = list()

	for i, val in enumerate(neuron_layers):
		
		# Initial Layer
		if(i == 0):
			layers.append(Dense(val, activation = 'relu', kernel_regularizer = regularizers.l2(regularizer_threshold), input_shape = (input_dims, 0)))
			if(dropout_threshold[i] != 0):
				layers.append(Dropout(dropout_threshold[i]))
		# Last Layer
		elif(i == len(neuron_layers) - 1):
			layers.append(Dense(output_dims, activation = 'relu', kernel_regularizer = regularizers.l2(regularizer_threshold)))
		# Intermediate hidden layers
		else:
			layers.append(Dense(val, activation = 'relu', kernel_regularizer = regularizers.l2(regularizer_threshold)))
			if(dropout_threshold[i] != 0):
				layers.append(Dropout(dropout_threshold[i]))

	model = Sequential(layers)

	return model
