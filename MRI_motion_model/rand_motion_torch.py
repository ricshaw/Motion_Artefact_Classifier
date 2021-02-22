import torch
import numpy as np
from rand_motion import rand_motion_3d, rand_motion_2d

class RandMotion(object):
    """Apply random motion artefact to the image."""

    def __init__(self, mode='2D', cfg=None):
        self.cfg = cfg
        self.mode = mode

    def __call__(self, image):
        if self.mode=='2D':
            image = rand_motion_2d(image)
        elif self.mode=='3D':
            image = rand_motion_3d(image)
        return image

class ToTensor(object):
    """Convert ndarrays to Tensors."""

    def __call__(self, image):
        image = np.expand_dims(image, axis=0)
        return torch.from_numpy(image)
