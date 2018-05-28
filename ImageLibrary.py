#This file is for functions dealing with extracting information from .nii files
#Uses NiBabel http://nipy.org/nibabel/gettingstarted.html
#Can use matplotlib to save to various other file types, such as .png

import os
import numpy as np
import nibabel as nib


class BrainImage:
	#This class represents a 3D brain image given in an NII file, and defines functions to operate on them
	def __init__(filename):
		self.filename = filename
		self.img = nib.load(filename)