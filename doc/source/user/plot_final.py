from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg

img = misc.face()
img_array = img / 255
img_array_transposed = np.transpose(img_array, (2, 0, 1))

U, s, Vt = linalg.svd(img_array_transposed)

Sigma = np.zeros((3, 768, 1024))
for j in range(3):
    np.fill_diagonal(Sigma[j, :, :], s[j, :])

k = 10

approx_img = U @ Sigma[..., :k] @ Vt[..., :k, :]
plt.imshow(np.transpose(approx_img, (1, 2, 0)))
