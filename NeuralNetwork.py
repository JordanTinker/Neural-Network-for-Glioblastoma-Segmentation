from keras.models import *
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization
import keras.optimizers
from keras.callbacks import ModelCheckpoint, History
from keras.utils import *
import keras.regularizers

import numpy as np

import time
import json
import pdb

from skimage import io

from sklearn.feature_extraction.image import extract_patches_2d


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


class NeuralNetwork:
	def __init__(self,
				type='basic',
				existing=None):
		#Constructor for model
		# TODO: determine what values need to be set here
		# TODO: What is the deal with num_filters per level?
		# TODO: What is the deal with kernel size?

		if existing is not None:
			self.model = load_existing_model(existing)
		elif type=='basic':
			self.compile_basic()
		elif type=='min':
			self.min_model()

		self.model.summary()
		#for layer in self.model.layers:
			#print("Layer Output shape {}".format(layer.output))
			#print(layer, layer.trainable)

		

	def compile_basic(self,
				epochs=5,
				num_channels=4,
				kernel_size=[7, 5, 5, 3],
				activation='relu',
				num_filters=[64,128,128,128]):
		self.epochs = epochs
		self.num_channels = num_channels
		# kernel_size is a list specifying the height and width of the 2D convolution window for each layer
		self.kernel_size = kernel_size
		self.activation = activation
		# Will be used for specifying the number of filters in each Convolutional Network
		# We have 4 values in it because we have 4 different types of images: FLAIR, T1, T2, T1 contrasted
		self.num_filters = num_filters
		self.num_classes = 4

		#Construct the model
		self.model = Sequential()

		# Begin adding layers

		# Layer 0
		self.model.add(Conv2D(self.num_filters[0],
						kernel_size[0],
						activation=self.activation,
						input_shape=(self.num_channels, 33, 33),
						data_format="channels_first"))
		self.model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1), data_format="channels_first"))
		#self.model.add(Dropout(0.5))

		# Layer 1
		self.model.add(Conv2D(self.num_filters[1],
						kernel_size[1],
						activation=self.activation,
						data_format="channels_first"))
		self.model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1), data_format="channels_first"))
		#self.model.add(Dropout(0.5))

		# Layer 2
		self.model.add(Conv2D(self.num_filters[2],
						kernel_size[2],
						activation=self.activation,
						data_format="channels_first"))
		self.model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1), data_format="channels_first"))
		#self.model.add(Dropout(0.5))

		# Layer 3
		self.model.add(Conv2D(self.num_filters[3],
						kernel_size[3],
						activation=self.activation,
						data_format="channels_first"))

		# No pooling necessary in this one because it is the last layer
		#self.model.add(Dropout(0.25))

		# flattening will get the output of the layers, flatten them to create a 1D vector
		# Source: https://stackoverflow.com/questions/43237124/role-of-flatten-in-keras
		self.model.add(Flatten())
		self.model.add(Dense(4))

		# "Often used as the final layer of a neural network classifier"
		# Source: https://en.wikipedia.org/wiki/Softmax_function#Neural_networks
		self.model.add(Activation('softmax'))

		# Stochastic gradient descent
		# Source: https://en.wikipedia.org/wiki/Stochastic_gradient_descent
		sgd = keras.optimizers.SGD(lr=.001, decay=.01, momentum=0.9)
		adam = keras.optimizers.Adam(lr=1e-6)


		# Using categorical crossentropy
		self.model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

	def min_model(self,
				epochs=10,
				num_channels=4,
				kernel_size=[7, 5, 5, 3],
				activation='relu',
				num_filters=[64,128,128,128]):
		self.epochs = epochs
		self.num_channels = num_channels
		# kernel_size is a list specifying the height and width of the 2D convolution window for each layer
		self.kernel_size = kernel_size
		self.activation = activation
		# Will be used for specifying the number of filters in each Convolutional Network
		# We have 4 values in it because we have 4 different types of images: FLAIR, T1, T2, T1 contrasted
		self.num_filters = num_filters
		self.num_classes = 4
		self.model = Sequential()
		self.model.add(Flatten(input_shape=(self.num_channels, 33, 33)))
		self.model.add(BatchNormalization())
		self.model.add(Dense(4, W_regularizer=keras.regularizers.l2(.02)))
		self.model.add(BatchNormalization())
		self.model.add(Activation('softmax'))

		self.model.compile(keras.optimizers.Adam(lr=1e-5), 'categorical_crossentropy', metrics=['accuracy'])


	def train_model(self, patch_list, labels_list, validation_data):
		#function to train model on data, will need to take in parameters for data

		data_formatting_start_time = time.time()
		print(labels_list.shape)
		labels_list = to_categorical(labels_list, num_classes=self.num_classes)
		print(labels_list.shape)
		vx = validation_data[0]
		vy = validation_data[1]
		vy = to_categorical(vy, num_classes=self.num_classes)
		validation_data = (vx,vy)

		#pdb.set_trace()
		# Create iterator from aggregation of elements from patch_list and labels_list
		# Source: https://stackoverflow.com/questions/31683959/the-zip-function-in-python-3

		#stuff_to_randomize = list(zip(patch_list, categorical_labels_list))
		#np.random.shuffle(stuff_to_randomize)

		#patch_list = np.array([stuff_to_randomize[i][0] for i in range(len(stuff_to_randomize))])
		#categorical_labels_list = np.array([stuff_to_randomize[i][1] for i in range(len(stuff_to_randomize))])

		# Checkpoint the model after each epoch
		checkpoint = ModelCheckpoint(filepath="checkpoint/bm_{epoch:02d}-{val_loss:.2f}.hdf5",
									monitor='val_loss',
									verbose=1)
		data_formatting_end_time = time.time()

		data_formatting_time = data_formatting_end_time - data_formatting_start_time
		print("data formatting time: " + str(data_formatting_time))

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


		fit_time_start = time.time()
		results = self.model.fit(patch_list,
					  labels_list,
					  epochs=self.epochs,
					  verbose=2,
					  validation_data=validation_data,
					  shuffle=True,
					  callbacks = [checkpoint])

		fit_time_end = time.time()
		fit_time = fit_time_end - fit_time_start
		print("fit time: " + str(fit_time))
		print("------------\n")
		print("results: " + str(results.history))

		self.save_model()
		self.save_architecture_and_weights()

	def predict_image(self, image):
		#function to evaluate an image and predict segmentation
		# TODO: will need another function for displaying segmented output

		# Read in the image with skimage, make all values float, reshape ndarray to mimic:
		#	1. Num of glioma classifications: Types 0, 1, 2, 3, 4
		#	2. Dimensions of image (240 x 240)

		time1 = time.time()
		#nd_array_image = io.imread(image, plugin='simpleitk').astype('float').reshape(5,240,240)
		nd_array_image = io.imread(image).astype('float').reshape(5,240,240)
		time_read = time.time() - time1
		print("Read image in {0}s".format(time_read))

		time2 = time.time()
		# Create patches
		patches_list = []
		for element in nd_array_image[:-1]:
			if np.amax(element) != 0:
				element = element / np.amax(element)
			# Generate patches for a slice that are 33x33
			patches_for_element = extract_patches_2d(element, (33, 33))
			patches_list.append(patches_for_element)

		packaged_patches = np.array(zip(
										np.array(patches_list[0]),
										np.array(patches_list[1]),
										np.array(patches_list[2]),
										np.array(patches_list[3])))
		time_patch = time.time() - time2
		print("made patches in {0}s".format(time_patch))
		time3 = time.time()

		predictions = self.model.predict_classes(packaged_patches)
		time_predict = time.time() - time3
		print("Predicted in {0}s".format(time_predict))
		reshaped_predictions = predictions.reshape(208,208)

		io.imshow(reshaped_predictions)

		return reshaped_predictions

	def save_model(self):
		# The save() function is from keras
		# It saves:
		#	the architecture of the model, thus allowing re-creation of the model
		#	the weights of the model
		#	the training configuration (loss, optimizer)
		#	the state of the optimizer, which allows it to resume training where it left off
		# Source: https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model
		self.model.save('current_model.h5')


	def save_architecture_and_weights(self):
		json_filename='current_model.json'
		weights_filename='current_weights.hdf5'
		json_string_representation = self.model.to_json()

		# The model and the weights are stored separately
		# The save_weights() method saves the weights of the models as an HDF5 file
		self.model.save_weights(weights_filename)
		with open(json_filename, 'w') as output_file:
			json.dump(json_string_representation, output_file)

def load_existing_model(filepath):
	# The load_model() function is from keras
	# It will load the model created by save() and will re-compile it
	# Source: https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model
	model = load_model(filepath)
	return model

def load_architecture_and_weights(self, json_filename, weights_file):
	with open(json_filename, 'r') as json_file:
		model = model_from_json(json_file.read())

	model.load_weights(weights_file)
	return model

def getClassFromPredict(prediction):
	if prediction[0][0] == 1.:
		return 0
	elif prediction[0][1] == 1.:
		return 1
	elif prediction[0][2] == 1.:
		return 2
	elif prediciton[0][3] == 1.:
		return 3
	elif prediction[0][4] == 1.:
		return 4