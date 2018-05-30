import sys
import numpy as np
import ImageLibrary as I
from Model import Model

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
		p = I.PatientData(f)
		presult = p.getNPatches(1500)
		patches = np.concatenate((patches, presult[0]))
		labels = np.concatenate((labels, presult[1]))

	labels = labels.flatten().astype(np.int_)

	return patches,labels


#usage: python TrainingShell.py <base path to data> train
if __name__ == '__main__':
	training_data = generateInput('t')
	#print('Training data shape is {0}, training labels shape is {1}'.format(training_data[0].shape, training_data[1].shape))
	validation_data = generateInput('v')
	#print('Validation data shape is {0}, validation labels shape is {1}'.format(validation_data[0].shape, validation_data[1].shape))

	model = Model()
	model.train_model(training_data[0], training_data[1], validation_data)
