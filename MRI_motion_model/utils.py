import numpy as np
import nibabel as nib
import os
import cv2
import matplotlib.pyplot as plt
from rand_motion import rand_motion_3d, rand_motion_2d

def normalise_image(image):
    if (image.max() - image.min()) < 1e-5:
        return image - image.min() + 1e-5
    else:
        return (image - image.min()) / (image.max() - image.min())

def load_nii_image(filename):
    image = nib.load(filename)
    image = np.asanyarray(image.dataobj).astype(np.float32)
    return normalise_image(image)

def load_png(filename):
    image = cv2.imread(filename, cv2.IMREAD_ANYDEPTH).astype(np.float32)
    return normalise_image(image)

def display_image(img, axs, row=0, cmap='gray'):
    axs[row,0].imshow(img[int(img.shape[0]/2),...],cmap)
    axs[row,1].imshow(img[:,int(img.shape[1]/2),:],cmap)
    axs[row,2].imshow(img[...,int(img.shape[0]/2)],cmap)
    return axs

def display_image_2d(img, axs, col=0, cmap='gray'):
    axs[col].imshow(img,cmap)
    return axs

def display_result_3d(img, output):
    diff = img - output
    h, axs = plt.subplots(3,3)
    axs = display_image(img, axs, row=0, cmap='gray')
    axs = display_image(output, axs, row=1, cmap='gray')
    axs = display_image(diff, axs, row=2, cmap='jet')
    axs[0,0].set_ylabel('Input')
    axs[1,0].set_ylabel('Output')
    axs[2,0].set_ylabel('Diff')
    return h

def display_result_2d(img, output):
    diff = img - output
    h, axs = plt.subplots(1,3)
    axs = display_image_2d(img, axs, col=0, cmap='gray')
    axs = display_image_2d(output, axs, col=1, cmap='gray')
    axs = display_image_2d(diff, axs, col=2, cmap='jet')
    axs[0].set_title('Input')
    axs[1].set_title('Output')
    axs[2].set_title('Diff')
    return h
