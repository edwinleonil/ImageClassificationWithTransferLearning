# write a function to equalize the histogram of an image with pytorch transforms
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


def equalize_hist(img):
    # use PIL to equalize the histogram of the image
    img_eq = Image.fromarray(np.array(img))
    img_eq = transforms.functional.equalize(img_eq)
    return img_eq

# test the function on an image
# load the image
img_path = r"C:\test_image.png"
img = Image.open(img_path)

# convert the image to a mode that supports the operation
img = img.convert('L')

# equalize the histogram
img_eq = equalize_hist(img)

# show the images side by side as subplots
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2)
ax[0].imshow(img, cmap="gray")
ax[1].imshow(img_eq, cmap="gray")
plt.show()