import sys
import numpy as np
from ImageLibrary import *
from NeuralNetwork import *
from keras.models import *
import pdb
import time

#parse the file to get the names of each set of data
def getFolderList(filename):
	with open(filename, 'r') as f:
		flist = f.readlines()

	flist = [x.strip() for x in flist]
	return flist

#mode is 't' or 'v'
#generate an array of patches and labels for the model
def generateInput(mode, patchesPerFile):
	flist = []
	if mode == 't':
		flist = getFolderList('traininglist.txt')
	elif mode == 'v':
		flist = getFolderList('validationlist.txt')
	else:
		print("Improper mode specified to generateInput.")
		exit(1)
	patches = np.array([]).reshape(0, 4, 33, 33)
	labels = np.array([]).reshape(0, 1)
	for f in flist:
		p = PatientData(f)
		presult = p.getNPatches(patchesPerFile)
		patches = np.concatenate((patches, presult[0]))
		labels = np.concatenate((labels, presult[1]))

	#labels = labels.flatten().astype(np.int_)
	labels = labels.astype(int)

	return patches,labels

#Generate a prediction image for a particular 2D brain slice
def runPrediction(name, sliceNum, outfile, network):
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
	seg_img = getHighlightedPNG(p.flair_data.data, seg_array, sliceNum)
	seg_img.save(outfile)
	predict_end = time.time()
	predict_time = predict_end - predict_start
	print("Highlighting finished finished in {0}s".format(predict_time))
	return seg_array

#train the model
def trainModel(model):
	training_data = generateInput('t', 320)
	print('Training data shape is {0}, training labels shape is {1}'.format(training_data[0].shape, training_data[1].shape))
	validation_data = generateInput('v', 320)
	print('Validation data shape is {0}, validation labels shape is {1}'.format(validation_data[0].shape, validation_data[1].shape))
	model.train_model(training_data[0], training_data[1], validation_data)

#For debugging, save the generated input patches to images
def saveInputPatches(patches):
	for i in range(patches[0].shape[0]):
		for chan in range(4):
			getPNGFromAnyPatch(patches[0][i][chan], "patches/{0}_chan{1}_L{2}.png".format(i,chan, patches[1][i]))


#Load the current model and generate all slices of predictions for one brain image
if __name__ == '__main__':
	model = NeuralNetwork(existing="current_model.h5")
	
	name = "Brats18_TCIA02_607_1"
	for z in range(155):
		runPrediction(name, z, "predictions/prediction_result_{0}_{1}.png".format(name, z), model)

	