import keras.models
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, History
import keras.utils

import numpy as np


# Useful links:
# https://adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks/
# http://cs231n.github.io/convolutional-networks/#layers
# http://jmlr.org/papers/volume15/srivastava14a.old/srivastava14a.pdf
# http://mipal.snu.ac.kr/images/1/16/Dropout_ACCV2016.pdf

# Useful functions from Keras Models:

# model.summary():		provides summary representation of the model

# model.get_config():	returns dictionary containing configuration of model
#	The model can be reinstantiated from the config:
#	config = model.get_config()
#	model = Sequential.from_config(config)

# model.get_weights():	Returns a list of all weight tensors, as Numpy arrays
# model.set_weights(weights):	Sets the values of the weights in the model

# model.to_json():		Returns representation of model as JSON string (does not include weights)
#	To reinitialize model (with new weights):
#	json = model.to_json()
#	model = model_from_json(json_string)

# To save the weights:
# model.save_weights(filepath):	Saves the weights of the model as an HDF5 file
# model.load_weights(filepath, by_name=False): loads the weights of the model from HDF5 file

# Notes about Sequential:
# Sequential is a linear stack of layers
# The model needs to know what input shape to expect, so the first layer needs to provide that info
# Note that further layers will do shape inference
# We can specify the input_shape to provide info to the first layer

# Compilation is the way you configure the learning process, done via compile() w/ 3 args:
#	1. optimizer: rmsprop or adagrad (list of optimizers here: https://keras.io/optimizers/)
#	2. loss: objective that the model will try to minimize (list here: https://keras.io/losses/)
#	3. metrics: metrics=['accuracy']

# Training: Models are trained using Numpy arrays and the fit() function

# Layers:
	# Core Layers of interest: Dense, Activation, Dropout, Flatten
		# Note that Dropout is used for prevention of overfitting
		# What it does is randomly drop units from the neural network during training
		# Source: http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf
	# Convolution Layers of interest: Conv2D

	# Pooling Layer: Normally, a Pooling layer is inserted between Convolutional Layers
	#	The idea is to reduce the size of the representation to reduce computation to reduce overfitting
	#	Source: http://cs231n.github.io/convolutional-networks/#layers


class Model:
	def __init__(self,
				epochs=5,
				num_channels=4,
				kernel_size = [7, 5, 5, 3],
				activation='relu',
				num_filters=[64,128,128,128]):
		#Constructor for model
		# TODO: determine what values need to be set here
		# TODO: What is the deal with num_filters per level?
		# TODO: What is the deal with kernel size?

		self.epochs = epochs
		self.num_channels = num_channels
		# kernel_size is a list specifying the height and width of the 2D convolution window for each layer
		self.kernel_size = kernel_size
		self.activation = activation
		# Will be used for specifying the number of filters in each Convolutional Network
		# We have 4 values in it because we have 4 different types of images: FLAIR, T1, T2, T1 contrasted
		self.num_filters = num_filters

		#Construct the model
		self.model = Sequential()

		# Begin adding layers

		# Layer 0
		self.model.add(Conv2D(self.num_filters[0],
						kernel_size[0],
						activation=self.activation,
						input_shape=(self.num_channels, 33, 33)))
		self.model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
		self.model.add(Dropout(0.5))

		# Layer 1
		self.model.add(Conv2D(self.num_filters[1],
						kernel_size[1],
						activation=self.activation))
		self.model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
		self.model.add(Dropout(0.5))

		# Layer 2
		self.model.add(Conv2D(self.num_filters[2],
						kernel_size[2],
						activation=self.activation))
		self.model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
		self.model.add(Dropout(0.5))

		# Layer 3
		self.model.add(Conv2D(self.num_filters[3],
						kernel_size[3],
						activation=self.activation))

		# No pooling necessary in this one because it is the last layer
		self.model.add(Dropout(0.25))

		# flattening will get the output of the layers, flatten them to create a 1D vector
		# Source: https://stackoverflow.com/questions/43237124/role-of-flatten-in-keras
		self.model.add(Flatten())
		self.model.add(Dense(5))

		# "Often used as the final layer of a neural network classifier"
		# Source: https://en.wikipedia.org/wiki/Softmax_function#Neural_networks
		self.model.add(Activation('softmax'))

		# Stochastic gradient descent
		# Source: https://en.wikipedia.org/wiki/Stochastic_gradient_descent

		# Using categorical crossentropy
		self.model.compile(loss='categorical_crossentropy', optimizer='sgd')


	def train_model(self, patch_list, labels_list, validation_data):
		#function to train model on data, will need to take in parameters for data

		categorical_labels_list = to_categorical(labels_list, 5)
		# Create iterator from aggregation of elements from patch_list and labels_list
		# Source: https://stackoverflow.com/questions/31683959/the-zip-function-in-python-3

		stuff_to_randomize = list(zip(patch_list, categorical_labels_list))
		np.random.shuffle(stuff_to_randomize)

		patch_list = np.array([stuff_to_randomize[i][0] for i in range(len(stuff_to_randomize))])
        categorical_labels_list = np.array([stuff_to_randomize[i][1] for i in range(len(stuff_to_randomize))])

		# Checkpoint the model after each epoch
		checkpoint = ModelCheckpoint(filepath="./checkpoint/bm_{epoch:02d}-{val_loss:.2f}.hdf5",
									monitor='val_loss',
									verbose=1)

		# The fit() method has these args that we care about:
		#	1. numpy array of training data
		#	2. numpy array of label data
		#	3. Number of samples per gradient update
		#	4. Number of epochs to train the model
		#	5. Verbosity level (0 = silent, 1 = progress bar, 2 = one line per epoch)
		#	6. Callbacks: List of callback instances
		#	7. validation_data: Tuple of (x_val, y_val), model will not be trained on this data

		# Note that we are not considering batch_size here

		# results is a History object
		# "History.history will show record of training
		# loss values and metrics values at successive epochs, as well as validation
		# loss values and validation metrics values"
		# Source: https://keras.io/callbacks/#history
		results = self.model.fit(patch_list,
					  categorical_labels_list,
					  epochs=self.epochs,
					  verbose=2,
					  validation_data=validation_data,
					  callbacks = [checkpoint])

	def predict_image(self):
		#function to evaluate an image and predict segmentation

	def save_state_of_model(self):
		json_filename='current_model.json'
		weights_filename='current_weights.hdf5'
		json_string_representation = self.model.to_json()

		# The model and the weights are stored separately
		# The save_weights() method saves the weights of the models as an HDF5 file
		self.model.save_weights(weights_filename)
		with open(json_filename, 'w') as output_file:
			json.dump(json_string_representation, output_file)
