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

	def getPNGFromSlice(self, numSlice):
		s = self.data[:, :, numSlice]
		im = Image.fromarray(s)
		#im.save("sample.png")


def floatArrayToL(src):
	max = np.amax(src)
	print (str(max))
	dst = np.ndarray(src.shape, dtype=np.uint8)

	return dst

if __name__ == '__main__':
	bi = BrainImage('sample.nii')
	bi.getPNGFromSlice(115)