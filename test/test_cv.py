import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import io
from skimage import transform
import cv2

img = io.imread("/home/cideep/Work/tensorflow/datasets/flower_photos/daisy/5547758_eea9edfd54_n.jpg")
print(img.shape)
res = transform.resize(img, [200, 200])
print(res.shape)
implot = plt.imshow(res)
plt.show()
