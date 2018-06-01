import sys
import numpy as np
from ImageLibrary import *
from NeuralNetwork import *
from keras.models import *
import pdb
import time

def getFolderList(filename):
	with open(filename, 'r') as f:
		flist = f.readlines()

	flist = [x.strip() for x in flist]
	return flist

#mode is 't' or 'v'
def generateInput(mode):
	flist = []
	if mode == 't':
		flist = getFolderList('traininglist.txt')
	elif mode == 'v':
		flist = getFolderList('validationlist.txt')
	else:
		print("Improper error specified to generateInput.")
		exit(1)
	patches = np.array([]).reshape(0, 4, 33, 33)
	labels = np.array([]).reshape(0, 1)
	for f in flist:
		p = PatientData(f)
		presult = p.getNPatches(1500)
		patches = np.concatenate((patches, presult[0]))
		labels = np.concatenate((labels, presult[1]))

	labels = labels.flatten().astype(np.int_)

	return patches,labels


if __name__ == '__main__':
	#training_data = generateInput('t')
	#print('Training data shape is {0}, training labels shape is {1}'.format(training_data[0].shape, training_data[1].shape))
	#validation_data = generateInput('v')
	#print('Validation data shape is {0}, validation labels shape is {1}'.format(validation_data[0].shape, validation_data[1].shape))

	#train
	#model = NeuralNetwork()
	#model.train_model(training_data[0], training_data[1], validation_data)

	#predict
	#pdb.set_trace()
	network = NeuralNetwork()
	network.model.load_weights("current_weights.hdf5")
	print("Loaded weights")

	data_start = time.time()
	print("Starting to get data at {0}".format(data_start))
	p = PatientData("Brats18_TCIA04_343_1")

	f = open("log.txt", 'w')
	range_start = 16
	while range_start < 224:
		range_end = range_start + 10
		if range_end > 224:
			range_end = 224
		predict_input = p.getPredictData(57, range_start, range_end)
		prediction = network.model.predict_classes(predict_input)
		f.write(str(prediction) + '\n')
		range_start += 10

	#predict_input = p.getPredictData(57)
	#p.flair_data.getPNGFromSlice(57, "samplepredict.png")
	data_end = time.time()
	data_time = data_end - data_start
	print("Got data in {0}s".format(data_time))
	
	predict_start = time.time()
	print("Starting predict at {0}".format(predict_start))
	#prediction = network.model.predict(predict_input)
	#prediction = network.predict_image("samplepredict.png")
	#parsed_prediction = getClassFromPredict(prediction)
	#print("Prediction:\n{0}".format(prediction))
	predict_end = time.time()
	predict_time = predict_end - predict_start
	print("Prediction finished in {0}s".format(predict_time))


	#seg_img = getHighlightedPNG(p.flair_data.data, segmentation, 65)
	#seg_img.save("sample_result.png")
