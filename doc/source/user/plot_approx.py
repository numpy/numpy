from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg

img = misc.face()
img_array = img / 255
img_gray = img_array @ [0.2126, 0.7152, 0.0722]

U, s, Vt = linalg.svd(img_gray)

Sigma = np.zeros((768, 1024))
for i in range(768):
    Sigma[i, i] = s[i]

k = 10

approx = U @ Sigma[:, :k] @ Vt[:k, :]
plt.imshow(approx, cmap="gray")
