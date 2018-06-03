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
		presult = p.getNPatches(150)
		patches = np.concatenate((patches, presult[0]))
		labels = np.concatenate((labels, presult[1]))

	labels = labels.flatten().astype(np.int_)

	return patches,labels

def runPrediction(name, sliceNum, outfile, network):
	#network = NeuralNetwork()
	#network.model.load_weights("current_weights.hdf5")
	#print("Loaded weights")

	data_start = time.time()
	print("Starting to get data at {0}".format(data_start))
	p = PatientData(name)

	seg_array = np.zeros((240,240), dtype='int')
	f = open("log.txt", 'w')
	for i in range(16, 224):
		predict_input = p.getPredictDataLine(sliceNum, i)
		prediction = network.model.predict_classes(predict_input)
		f.write(str(prediction) + '\n')
		for j in range(16, 224):
			seg_array[i][j] = prediction[j-16]

	f.close()
	#predict_input = p.getPredictData(57)
	#p.flair_data.getPNGFromSlice(57, "samplepredict.png")
	data_end = time.time()
	data_time = data_end - data_start
	print("Got data in {0}s".format(data_time))
	
	predict_start = time.time()
	print("Starting segmentation highlighting")
	seg_img = getHighlightedPNG(p.flair_data.data, seg_array.T, sliceNum)
	seg_img.save(outfile)
	predict_end = time.time()
	predict_time = predict_end - predict_start
	print("Highlighting finished finished in {0}s".format(predict_time))
	return seg_array


if __name__ == '__main__':
	training_data = generateInput('t')
	print('Training data shape is {0}, training labels shape is {1}'.format(training_data[0].shape, training_data[1].shape))
	validation_data = generateInput('v')
	print('Validation data shape is {0}, validation labels shape is {1}'.format(validation_data[0].shape, validation_data[1].shape))
	#train
	model = NeuralNetwork()
	model.train_model(training_data[0], training_data[1], validation_data)

	#predict
	#pdb.set_trace()
	runPrediction("Brats18_2013_2_1", 106, "prediction_result.png", model)
	

	#seg_img = getHighlightedPNG(p.flair_data.data, segmentation, 65)
	#seg_img.save("sample_result.png")
