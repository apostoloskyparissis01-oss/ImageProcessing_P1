import cv2 
import numpy as np
from skimage import exposure
import matplotlib.pyplot as plt

img1 = cv2.imread('images/cameraman.tif', cv2.IMREAD_GRAYSCALE)
img1_dark = (img1 *0.3).astype(np.uint8)
img2 = cv2.imread('images/pirate.tif', cv2.IMREAD_GRAYSCALE)
img3 = cv2.imread('images/Fig0819(a).tif', cv2.IMREAD_GRAYSCALE)
img4 = cv2.imread('images/salt-and-pepper-noise.png',cv2.IMREAD_GRAYSCALE)

def show_images(imgs, results, titles):
    n = len(results)
    fig, axes = plt.subplots(n, 2, figsize=(12, n*9))
    axes[0][0].set_title('Πριν') 
    axes[0][1].set_title('Μετά')


    for i in range(n):
        fig.text(0.5, 1-(i+0.5)/n, titles[i], ha='center', va='center', rotation='horizontal', fontsize=12,fontweight = 'bold')
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

def img_lap(img,c):
    lap = cv2.Laplacian(img, cv2.CV_64F)
    lap = lap / np.abs(lap).max() * 25
    result = img.astype(np.float64) + c * lap
    result = result -result.min()
    result = result / result.max() * 255
    result = result.astype(np.uint8)
    return result 

def fourier_no_center(img):
    F = np.fft.fft2(img.astype(np.float64))
    magnitude = np.abs(F)
    magnitude = np.log1p(magnitude)
    magnitude = magnitude / magnitude.max() * 255
    return magnitude.astype(np.uint8)

def fourier_center(img):
    F = np.fft.fft2(img.astype(np.float64))
    Fshift = np.fft.fftshift(F)
    magnitude = np.abs(Fshift)
    magnitude = np.log1p(magnitude)
    magnitude = magnitude / magnitude.max() * 255
    return magnitude.astype(np.uint8)

def freq_f(img):

    mask = np.ones((3,3), np.float64) / 9

    F_img = np.fft.fft2(img.astype(np.float64))
    F_mask = np.fft.fft2(mask, s=img.shape)
    
    result = F_img * F_mask

    freq = np.real(np.fft.ifft2(result))
    freq = np.clip(freq, 0, 255).astype(np.uint8)
    return freq

def rotate_f(img):
    F = np.fft.fft2(img.astype(np.float64))
    Fshift = np.fft.fftshift(F)
    magnitude = np.abs(Fshift)
    magnitude = np.log1p(magnitude)
    magnitude = magnitude / magnitude.max() * 255
    return magnitude.astype(np.uint8)

def low_pass_filter(img,D0):
    F = np.fft.fft2(img.astype(np.float64))
    Fshift = np.fft.fftshift(F)

    h,w = img.shape
    cy,cx = h//2, w//2

    mask = np.zeros((h,w), np.float64) 

    for y in range(h):
        for x in range(w):
            D = np.sqrt((x-cx)**2 + (y-cy)**2)
            if D <= D0:
                mask[y,x] = 1

    Fshift_filtered = Fshift * mask
    F_back = np.fft.ifftshift(Fshift_filtered)
    result = np.real(np.fft.ifft2(F_back))
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


def high_pass_filter(img):
    F = np.fft.fft2(img.astype(np.float64))
    Fshift = np.fft.fftshift(F)

    h,w = img.shape
    cy,cx = h//2, w//2

    mask = np.zeros((h,w), np.float64) 

    for y in range(h):
        for x in range(w):
            D = np.sqrt((x-cx)**2 + (y-cy)**2)
            if D <= 30:
                mask[y,x] = 0

    Fshift_filtered = Fshift * mask
    F_back = np.fft.ifftshift(Fshift_filtered)
    result = np.real(np.fft.ifft2(F_back))
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


def show_kernel(img):
    h,w = img.shape
    cy,cx = h//2, w//2

    mask = np.zeros((h,w), np.float64) 

    for y in range(h):
        for x in range(w):
            D = np.sqrt((x-cx)**2 + (y-cy)**2)
            if D <= 30:
                mask[y,x] = 1
    
    kernel = np.real(np.fft.ifft2(np.fft.ifftshift(mask)))
    kernel = np.fft.fftshift(kernel)
    
    kernel = kernel - kernel.min()
    kernel = kernel / kernel.max() * 255
    return kernel.astype(np.uint8)

F = np.fft.fft2(img1.astype(np.float64))
Fshift = np.fft.fftshift(F)
magnitude = np.abs(Fshift)
magnitude = np.log1p(magnitude)
magnitude = magnitude / magnitude.max() * 255
fourier_img1 = magnitude

imgs = [img1, img1, img1, img1, img1, img1, img1_dark,img1_dark,img1,img1,img1,img4,img4,img3,img3,img2,img2,img2,img2,fourier_img1,img1,img1,img1,img2,img2]
results = [img_negative(img1),img_log(img1,1), img_log(img1,20),img_gamma(img1,0.4),img_gamma(img1,1),img_gamma(img1,2.5),img_eq(img1_dark),img_matched(img1_dark,img2),img_blur(img1,3),img_blur(img1,9),img_blur(img1,15),img_median(img4,3),img_median(img4,5),img_lap(img3,1),img_lap(img3,-1),fourier_no_center(img2),fourier_center(img2),img_blur(img2,3),freq_f(img2),rotate_f(cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)),low_pass_filter(img1,30),low_pass_filter(img1,60),low_pass_filter(img1,120), high_pass_filter(img2),show_kernel(img2)]
titles = ["Αρνητικό Εικονάς", "Λογαριθμικός μετασχηματισμό c=1", "Λογαριθμικός μετασχηματισμό c=20", " Μετασχηματισμός γ=0.4"," Μετασχηματισμός γ=1"," Μετασχηματισμός γ=2.5","Εξίσωση Ιστογράμματος","Ταίριασμα ιστογράμματος","averaging filter 3x3","averaging filter 9x9","averaging filter 15x15","median filter 3x3","median filter 5x5","Φίλτρο οξύτητας: Λαπλασιανή c=1","Φίλτρο οξύτητας: Λαπλασιανή c=-1"," Μετασχηματισμός Fourier no center"," Μετασχηματισμός Fourier center","Averaging filter 3x3","Averaging filter 3x3 fourier * fourier image","90° rotation","Low-Pass Filter D0=30","Low-Pass Filter D0=60","Low-Pass Filter D0=120","High-Pass Filter D0=30","Kernel of Low_pass_filter D0=30"]


show_images(imgs,results,titles)





