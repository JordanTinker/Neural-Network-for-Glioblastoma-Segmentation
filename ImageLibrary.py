#This file is for functions dealing with extracting information from .nii files
#Uses NiBabel http://nipy.org/nibabel/gettingstarted.html
#Can use matplotlib to save to various other file types, such as .png

import os
import numpy as np
import nibabel as nib
from PIL import Image


class BrainImage:
	#This class represents a 3D brain image given in an NII file, and defines functions to operate on them
	def __init__(self, filename):
		self.filename = filename
		self.img = nib.load(filename)
		self.data = self.img.get_data()

	def printInfo(self):
		print("Printing information for {0}".format(self.filename))
		obj = self.img.dataobj
		isArray = nib.is_proxy(obj)
		print("Proxy: {0}".format(str(isArray)))
		data = self.img.get_data()
		print("Shape:\n{0}".format(str(data.shape)))
		print("Array:\n{0}".format(data[:, :, 115]))

	def getPNGFromSlice(self, numSlice, outfile):
		s = self.data[:, :, numSlice].T
		for x in np.nditer(s, op_flags=['readwrite']):
			x[...] = np.uint8(np.float64(x/1000)*255)
		im = Image.fromarray(s.astype(np.uint8), mode='L')
		im.save(outfile)

	#return a Numpy array containing a 33x33 patch centered around the x,y coordinate in the z plane
	def getPatch(self, x, y, z):
		return self.data[(x-16):(x+17), (y-16):(y+17), z]

	#save a patch as a PNG
	def getPNGFromPatch(self, x, y, z, outfile):
		patch = self.getPatch(x, y, z)
		for x in np.nditer(patch, op_flags=['readwrite']):
			x[...] = np.uint8(np.float64(x/1000)*255)
		im = Image.fromarray(patch.astype(np.uint8), mode='L')
		im.save(outfile)

	def getValueAt(self, x, y, z):
		return self.data[x, y, z]


class PatientData:

	def __init__(self, name):
		self.name = name
		path = "data/" + name + "/" + name
		print(path)
		self.flair_data = BrainImage(path + '_flair.nii.gz')
		self.t1_data = BrainImage(path + '_t1.nii.gz')
		self.t1ce_data = BrainImage(path + '_t1ce.nii.gz')
		self.t2_data = BrainImage(path + '_t2.nii.gz')
		self.groundtruth = BrainImage(path + '_seg.nii.gz')

	def getGroundTruth(self, x, y, z):
		return self.groundtruth.getValueAt(x, y, z)

	def getNPatches(self, n):
		numlist = []
		for i in range(n):
			numlist.append((np.random.randint(16, high=224), np.random.randint(16, high=224), np.random.randint(0, high=155)))
		#print("Numlist length is {0}".format(len(numlist)))
		patchlist = []
		for i in range(n):
			coords = numlist[i]
			flair_patch = self.flair_data.getPatch(coords[0], coords[1], coords[2])
			t1_patch = self.t1_data.getPatch(coords[0], coords[1], coords[2])
			t1ce_patch = self.t1ce_data.getPatch(coords[0], coords[1], coords[2])
			t2_patch = self.t2_data.getPatch(coords[0], coords[1], coords[2])
			stacked = np.stack((flair_patch, t1_patch, t1ce_patch, t2_patch))
			patchlist.append(stacked)
		#print("Patchlist length is {0} before validation".format(len(patchlist)))
		valid_patches = []
		labels = []
		for i in range(n):
			if validatePatch(patchlist[i][0]):
				valid_patches.append(patchlist[i])
				coords = numlist[i]
				labels.append(self.groundtruth.getValueAt(coords[0], coords[1], coords[2]))
		#print("There are {0} valid patches".format(len(valid_patches)))
		#print("There are {0} labels".format(len(labels)))

		result_patches = np.array([]).reshape(0, 4, 33, 33)
		result_labels = np.array([]).reshape(0, 1)
		for e in valid_patches:
			e2 = e.reshape(1, 4, 33, 33)
			result_patches = np.vstack((result_patches, e2))
		for e in labels:
			e2 = np.array([e]).reshape(1, 1)
			result_labels = np.vstack((result_labels, e2))

		#print("Shape of result_patches: {0} Shape of labels: {1}".format(result_patches.shape, result_labels.shape))

		return (result_patches, result_labels)

def validatePatch(patch):
	# A patch is valid if less than 20% of the pixels in the image are value 0 (black background)
	totalZeros = 0
	for x in np.nditer(patch):
		if x == 0:
			totalZeros += 1
	if (totalZeros > 217):
		return False
	else:
		return True


if __name__ == '__main__':
	#bi = BrainImage('sample.nii.gz')
	#bi.getPNGFromPatch(78, 64, 106, "samplepatch.png")
	p = PatientData("as", "Brats18_2013_2_1")
	result = p.getNPatches(2000)
	#p.flair_data.getPNGFromSlice(106, "sample1.png")
	#print(str(p.getNPatches(50)))