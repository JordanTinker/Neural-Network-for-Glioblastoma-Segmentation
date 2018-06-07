# CS168Project
Repository for CS 168 project, Computational Methods for Medical Imaging, with Professor Fabien Scalzo

## Files
NeuralNetwork.py - main file for Keras model, includes function for compiling model, training, and evaluating

ImageLibrary.py - main file for manipulating NifTi images (.nii) and extracting data for use with model

TrainingShell.py - main file for generating input and running the model to train or predict on images.

traininglist.txt - the list of which files to use for training data

validationlist.txt - the list of which files to use for validation data

current_model.h5 - the saved form of the trained model. The architecture and weights can both be loaded from this

current_model.json and current_weights.hdf5 - The current model and weights saved separately. Not needed with current_model.h5

## Dependencies:
The following dependencies are needed to run this program:
Python 3.5
Keras
Tensorflow
NiBabel
SciPy
Pillow
h5py
scikit-image
scikit-learn

## To run this program:
One folder of sample data is provided. The entire dataset is too large to use on github. The current model can be loaded in order to generate a prediction on the sample data. The main function of 'TrainingShell.py' is set up to generate a prediction over the entire 3D brain image, slice by slice. Running this can take hours. You can modify this main file to only predict on a single 2D slice with a specified z coordinate if you wish.

The main function of 'ImageLibrary.py' is set up to generate png images for each slice of the 3D image using the ground truth data. This can also be modified to only do one slice as well. This runs much quicker than TrainingShell.

You may need to create directories called 'groundtruth' and 'predictions' in order for the program to properly save the images.
