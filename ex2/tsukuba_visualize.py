# Author: Stefan Haller <stefan.haller@iwr.uni-heidelberg.de>

import numpy
from PIL import Image


def to_image(labeling, dimensions):
    width, height = dimensions
    data = numpy.asarray(labeling).reshape(height, width) / 15
    return Image.fromarray(numpy.uint8(data * 255))

# You can use `to_image(labeling, dimensions).show()` to display the resulting
# image. Use `.save(filename)` to save it to disk.
