import cv2 
import numpy as np
from scipy import ndimage
from skimage import exposure, filters
import matplotlib.pyplot as plt

img1 = cv2.imread('images/cameraman.tif', cv2.IMREAD_GRAYSCALE)
img1_dark = (img1 *0.3).astype(np.uint8)
img2 = cv2.imread('images/pirate.tif', cv2.IMREAD_GRAYSCALE)


def show_images(imgs, results, titles):
    n = len(results)

    fig, axes = plt.subplots(n, 2, figsize=(12, n*9))
    axes[0][0].set_title('Πριν') 
    axes[0][1].set_title('Μετά')


    for i in range(n):
        fig.text(0.5, 1-(i+0.5)/n, titles[i], ha='center', va='center', rotation='horizontal', fontsize=12)
        axes[i][0].imshow(imgs[i], cmap='gray')
        axes[i][0].axis('off') 

        axes[i][1].imshow(results[i], cmap='gray')
        axes[i][1].axis('off')

    
    plt.tight_layout()
    plt.savefig('results.png', bbox_inches='tight', dpi=300)
    plt.show()

    if "Ταίριασμα ιστογράμματος":
        fig2, axes2 = plt.subplots(1,1, figsize=(6, 6))
        fig2.suptitle('Reference for histogram matching',fontsize=14)
        axes2.imshow(img2,cmap='gray')
        axes2.axis('off')
        plt.show()


def img_negative(img):
    return cv2.bitwise_not(img)

def img_log(img, c):
    return  exposure.adjust_log(img, c)

def img_gamma(img, g):
    return exposure.adjust_gamma(img,g)

def img_eq(img):
    return cv2.equalizeHist(img)

def img_matched(img_s, img_r):
    return exposure.match_histograms(img_s, img_r)

def img_blur(img,ksize):
    return cv2.blur(img,(ksize,ksize))

def img_median(img,ksize):
    return cv2.medianBlur(img,ksize)

imgs = [img1, img1, img1, img1, img1, img1, img1_dark,img1_dark,img1,img1,img1,img1,img1]
results = [img_negative(img1),img_log(img1,1), img_log(img1,20),img_gamma(img1,0.4),img_gamma(img1,1),img_gamma(img1,2.5),img_eq(img1_dark),img_matched(img1_dark,img2),img_blur(img1,3),img_blur(img1,9),img_blur(img1,15),img_median(img1,3),img_median(img1,5)]
titles = ["Αρνητικό Εικονάς", "Λογαριθμικός μετασχηματισμό c=1", "Λογαριθμικός μετασχηματισμό c=20", " Μετασχηματισμός γ=0.4"," Μετασχηματισμός γ=1"," Μετασχηματισμός γ=2.5","Εξίσωση Ιστογράμματος","Ταίριασμα ιστογράμματος","averaging filter 3x3","averaging filter 9x9","averaging filter 15x15","median filter 3x3","median filter 5x5"]


show_images(imgs,results,titles)





