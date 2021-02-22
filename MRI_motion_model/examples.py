import numpy as np
import nibabel as nib
import os
import cv2
import matplotlib.pyplot as plt
import utils
from rand_motion import rand_motion_3d, rand_motion_2d

cfg = {'lambda_large': 2,
       'lambda_small': 3,
       'max_large': 5,
       'max_small': 10,
       'angles_stddev_large': 10.,
       'angles_stddev_small': 3.,
       'trans_stddev_large': 10.,
       'trans_stddev_small': 2.,
       'min_kspace': 0.3,
       'max_kspace': 0.7,
       'pad_width': 20,
       'trajectory':'cartesian',
       'debug':True}


def example_2d():
    # Load 2D image
    filename = './data/sample_2d.png'
    img = utils.load_png(filename)
    print('Input:', img.shape)

    # Randon motion 2D
    output = rand_motion_2d(img, cfg=cfg)
    print('Output:', output.shape)
    h = utils.display_result_2d(img, output)
    plt.show()
    return


def example_3d():
    # Load 3D image
    filename = './data/sample_3d.nii.gz'
    img = utils.load_nii_image(filename)
    print('Input:', img.shape)

    # Random motion 3D
    output = rand_motion_3d(img, cfg=cfg)
    print('Output:', output.shape)
    h = utils.display_result_3d(img, output)
    plt.show()
    return


def torch_example_2d():
    import torch
    import torchvision
    from torchvision import transforms
    from rand_motion_torch import RandMotion, ToTensor

    # Load 2D image
    filename = './data/sample_2d.png'
    img = utils.load_png(filename)
    print('Input:', img.shape)

    transform = transforms.Compose([RandMotion(mode='2D', cfg=cfg),
                                    ToTensor(),
                                   ])

    transformed_sample = transform(img)
    print('Output:', transformed_sample.shape)
    h = utils.display_result_2d(img, transformed_sample.numpy()[0,...])
    plt.show()
    return


def torch_example_3d():
    import torch
    import torchvision
    from torchvision import transforms
    from rand_motion_torch import RandMotion, ToTensor

    # Load 3D image
    filename = './data/sample_3d.nii.gz'
    img = utils.load_nii_image(filename)
    print('Input:', img.shape)

    transform = transforms.Compose([RandMotion(mode='3D', cfg=cfg),
                                    ToTensor(),
                                   ])

    transformed_sample = transform(img)
    print('Output:', transformed_sample.shape)
    h = utils.display_result_3d(img, transformed_sample.numpy()[0,...])
    plt.show()
    return


example_2d()
#example_3d()
#torch_example_2d()
#torch_example_3d()
