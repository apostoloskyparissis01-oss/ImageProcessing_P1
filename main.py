import cv2 
import numpy as np
from scipy import ndimage
from skimage import exposure, filters
import matplotlib.pyplot as plt

img = cv2.imread('images/cameraman.tif', cv2.IMREAD_GRAYSCALE)

img_processed = exposure.adjust_gamma(img, 1.5)

fig, axes = plt.subplots(1, 2, figsize=(12,6))

fig.text(0.02, 0.5, 'Gamma Correction', va='center', rotation='vertical', fontsize=12)

axes[0].imshow(img, cmap='gray')
axes[0].set_title('Πριν') 
axes[0].axis('off') 

axes[1].imshow(img_processed, cmap='gray')
axes[1].set_title('Μετά')
axes[1].axis('off')

plt.show()