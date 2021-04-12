from scipy import misc
import matplotlib.pyplot as plt
import numpy as np

img = misc.face()
img_array = img / 255
img_gray = img_array @ [0.2126, 0.7152, 0.0722]
plt.imshow(img_gray, cmap="gray")
