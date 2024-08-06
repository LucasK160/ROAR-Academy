#exercise 2
from matplotlib import image
from matplotlib import pyplot as plt
import os
import numpy as np

#resize function (not created by me):
def resize_image(image, new_height, new_width):
    old_height, old_width, channels = image.shape
    resized_image = np.zeros((new_height, new_width, channels))

    row_ratio = old_height / new_height
    col_ratio = old_width / new_width

    for i in range(new_height):
        for j in range(new_width):
            old_i = int(i * row_ratio)
            old_j = int(j * col_ratio)
            resized_image[i, j] = image[old_i, old_j]

    return resized_image


path = os.path.dirname(os.path.abspath(__file__))
filename = path + '/' + 'lenna.bmp'
data = image.imread(filename)

path1 = os.path.dirname(os.path.abspath(__file__))
filename1 = path1 + '/' + 'flag.png'
data1 = image.imread(filename1)

#flag was black and didn't know why, chatgpt told me to do this
if data.max() > 1:
    data = data / 255.0
if data1.max() > 1:
    data1 = data1 / 255.0

new_height, new_width = 60,100

data1=resize_image(data1, new_height, new_width)


#not entirely sure why, but needed this to ignore the 4th channel which the flag image has
data1_rgb = data1[:, :, :3]


data = np.copy(data)


flag_height, flag_width, _ = data1_rgb.shape

data[0:flag_height, -flag_width:] = data1_rgb
plt.imshow(data)
plt.show()


