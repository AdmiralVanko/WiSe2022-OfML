#!/usr/bin/env python
#
# Author: Stefan Haller <stefan.haller@iwr.uni-heidelberg.de>

# Before using this you have to install the Python Pillow and numpy libraries.

from PIL import Image
import numpy as np

# To load an image:
img = Image.open('seam_carving.jpg')

# You can show the image in a new window:
img.show()

# You can create an numpy array out of an image. The underlying numpy data type
# (dtype) will be chosen atomatically. For a normal RGB image you will get a
# matrix of the shape `height x width x 3` of dtype np.uint8.
data = np.asarray(img)
row = 3
column = 2
pixel = data[row, column]
r, g, b = pixel

# Note that if you want to do arithmetics on the data, you might want to change
# the datatype as and np.uint8 will overflow.
# For example: `np.uint(3) - np.uint(4) == 255`.
print(np.abs(data[row-1, column].astype(np.float64) - data[row+1, column].astype(np.float64)).sum())

# You create an image out of numpy array in the following way:
img = Image.fromarray(data)
img.save('out.png')
