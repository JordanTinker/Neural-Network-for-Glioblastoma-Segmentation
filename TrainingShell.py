import sys
import numpy as np
from ImageLibrary import *
from NeuralNetwork import *
from keras.models import *
import pdb

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
	pdb.set_trace()
	network.model.load_weights("current_weights.hdf5")
	print("Loaded weights")
	segmentation = network.predict_image("data/Brats18_TCIA04_343_1/Brats18_TCIA04_343_1_flair.nii.gz")
	p = PatientData("Brats18_TCIA04_343_1")
	seg_img = getHighlightedPNG(p.flair_data.data, segmentation, 65)
	seg_img.save("sample_result.png")
