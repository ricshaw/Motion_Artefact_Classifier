import numpy as np
import scipy.ndimage
import SimpleITK as sitk
from scipy.linalg import logm, expm
from numpy.linalg import inv
import matplotlib.pyplot as plt
import skimage.draw
from skimage.draw import disk
import sys
import gc
import random

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

def get_default_config():
    """Define default configuration.
    Parameters:
        lambda_large: Poisson dist parameter for large movements
        lambda_small: Poisson dist parameter for small movements
        max_large: maximum number of large movements
        max_small: maximum number of small movements
        angles_stddev_large: standard deviation of large angles (degrees)
        angles_stddev_small: standard deviation of small angles (degrees)
        trans_stddev_large: standard deviation of large translations (voxels)
        trans_stddev_small: standard deviation of small translations (voxels)
        min_kspace: earliest part of the kspace to sample movements (0->1) 
        max_kspace: latest part of the kspace to sample movements (0->1)
        pad_width: edge padding to prevent edge effects when transforming
        trajectory: k-space scanning trajectory
        debug: for debugging information
    """
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
    return cfg


def normalise_image(image):
    """Normalise image from 0 to 1."""
    if (image.max() - image.min()) < 1e-5:
        return image - image.min() + 1e-5
    else:
        return (image - image.min()) / (image.max() - image.min())


def getRotationMatrix2D(a):
    """2D rotation matrix."""
    return np.array([[np.cos(a), -np.sin(a)],
                     [np.sin(a),  np.cos(a)]], np.float32)


def getRotationMatrix3D(angles):
    """3D rotation matrix."""
    ax, ay, az = angles[0], angles[1], angles[2]
    Rx = np.array([[1,          0,           0],
                   [0, np.cos(ax), -np.sin(ax)],
                   [0, np.sin(ax),  np.cos(ax)]], np.float32)
    Ry = np.array([[np.cos(ay),  0, np.sin(ay)],
                   [0,           1,          0],
                   [-np.sin(ay), 0, np.cos(ay)]], np.float32)
    Rz = np.array([[np.cos(az), -np.sin(az), 0],
                   [np.sin(az),  np.cos(az), 0],
                   [0,                    0, 1]], np.float32)
    return np.dot(Rz,np.dot(Ry,Rx))


def getAffineMatrixITK(R, t):
    """4x4 3D affine matrix."""
    A = np.eye(4)
    A[:3,:3] = R
    A[:3,3] = t.T
    A = inv(A)
    A[3,:] = [0.,0.,0.,1.]
    return A


def getAffine2DMatrixITK(R, t):
    """3x3 2D affine matrix."""
    A = np.eye(3)
    A[:2,:2] = R
    A[:2,2] = t.T
    A = inv(A)
    A[2,:] = [0.,0.,1.]
    return A


def affine3DTranformITK(image_3d, A_itk):
    """Resample 3D image with ITK affine transform"""
    image_3d = sitk.GetImageFromArray(image_3d)
    A = sitk.AffineTransform(3)
    t = A_itk[:,3]
    R = A_itk[:3,:3]
    A.SetTranslation((t[0],t[1],t[2]))
    A.SetMatrix(R.ravel())
    centre = (0.5*image_3d.GetSize()[0], 0.5*image_3d.GetSize()[1], 0.5*image_3d.GetSize()[2])
    A.SetCenter(centre)
    interpolator = sitk.sitkBSpline
    image_3d = sitk.Resample(image_3d, image_3d.GetSize(), A, interpolator, [0,0,0])
    return sitk.GetArrayFromImage(image_3d)


def affine2DTranformITK(image_2d, A_itk):
    """Resample 2D image with ITK affine transform"""
    image_2d = sitk.GetImageFromArray(image_2d)
    A = sitk.AffineTransform(2)
    t = A_itk[:,2]
    R = A_itk[:2,:2]
    A.SetTranslation((t[0],t[1]))
    A.SetMatrix(R.ravel())
    centre = (0.5*image_2d.GetSize()[0], 0.5*image_2d.GetSize()[1])
    A.SetCenter(centre)
    interpolator = sitk.sitkBSpline
    image_2d = sitk.Resample(image_2d, image_2d.GetSize(), A, interpolator, [0,0])
    return sitk.GetArrayFromImage(image_2d)


def fft(image):
    """N-dimensional FFT."""
    return np.fft.fftshift(np.fft.fftn(image))


def ifft(F):
    """N-dimensional inverse FFT."""
    return np.fft.ifftn(np.fft.ifftshift(F))


def fft2D(image):
    """2D FFT."""
    return np.fft.fftshift(np.fft.fft2(image, axes=(0,1)))


def ifft2D(F):
    """2D inverse FFT."""
    return np.fft.ifft2(np.fft.ifftshift(F), axes=(0,1))


def getRotations(angles, mode=None):
    """Get list of rotation matrices from angles."""
    rotations = []
    for i in range(angles.shape[1]):
        if mode=='2D':
            R = getRotationMatrix2D(angles[0,i])
        elif mode=='3D':
            R = getRotationMatrix3D(angles[:,i])
        else:
            sys.exit('Specify 2D or 3D rotations')
        rotations.append(R)
    return rotations


def getMasks3D(image_3d, times, num_movements):
    """Get a list of 3D k-space masks."""
    num_k = image_3d.size
    movement_k = np.floor(num_k * times)
    masks = []
    mask = np.arange(0,movement_k[0],1,int)
    masks.append(mask)
    for i in range(num_movements-1):
        mask = np.arange(movement_k[i],movement_k[i+1],1,int)
        masks.append(mask)
    mask = np.arange(movement_k[num_movements-1],num_k,1,int)
    masks.append(mask)
    return masks


def getMasks2D(image_2d, times, num_movements):
    """Get a list of 2D k-space masks."""
    masks = []
    mask = np.zeros_like(image_2d)
    mask[0:times[0],:] = 1
    masks.append(mask)
    for i in range(num_movements-1):
        mask = np.zeros_like(image_2d)
        mask[times[i]:times[i+1],:] = 1
        masks.append(mask)
    mask = np.zeros_like(image_2d)
    mask[times[-1]::,:] = 1
    masks.append(mask)
    return masks


def getSpiralMasks2D(image_2d, times, num_movements):
    """Get a list of 2D k-space spiral masks."""
    masks = []
    mask = np.zeros_like(image_2d)
    rows, cols = image_2d.shape[:2]
    crow   = int(0.5*rows)
    ccol   = int(0.5*cols)
    rr, cc = disk((crow, ccol), times[0])
    mask[rr, cc] = 1
    masks.append(mask)
    for i in range(num_movements-1):
        mask = np.zeros_like(image_2d)
        rr, cc = disk((crow, ccol), times[i+1])
        mask[rr, cc] = 1
        rr, cc = disk((crow, ccol), times[i])
        mask[rr, cc] = 0
        masks.append(mask)
    mask = np.ones_like(image_2d)
    rr, cc = disk((crow, ccol), times[-1])
    mask[rr, cc] = 0
    masks.append(mask)
    return masks


def computeWeights3D(masks, image_3d, times, num_movements):
    """Get weights for each 3D movement transform."""
    num_k = image_3d.size
    movement_k = np.floor(num_k * times).astype(int)
    #print('k-space elements:', movement_k)
    crow   = int(0.5*image_3d.shape[0])
    ccol   = int(0.5*image_3d.shape[1])
    cdepth = int(0.5*image_3d.shape[2])
    weights = np.zeros((num_movements+1), dtype=np.float32)
    xx = np.arange(-ccol,ccol+1,1)
    yy = np.arange(-crow,crow+1,1)
    zz = np.arange(-cdepth,cdepth+1,1)
    X, Y, Z = np.meshgrid(xx,yy,zz)
    r = np.exp(-(np.square(X) + np.square(Y) + np.square(Z)), dtype=np.float32)
    for i in range(num_movements+1):
        mask = masks[i]
        r_masked = r[np.unravel_index(mask, r.shape, 'F')]
        weights[i] = np.sum(r_masked) / np.sum(r)
    weights = weights / np.sum(weights)
    #print('Weights:', weights)
    return weights


def computeWeights2D(masks, image_2d, times, num_movements):
    """Get weights for each 2D movement."""
    num_k = image_2d.size
    rows, cols = image_2d.shape[:2]
    crow   = int(0.5*rows) - 0.5
    ccol   = int(0.5*cols) - 0.5
    weights = np.zeros((num_movements+1), dtype=np.float32)
    xx = np.linspace(-ccol, ccol, cols)
    yy = np.linspace(-crow, crow, rows)
    X, Y = np.meshgrid(xx,yy)
    r = np.exp(-(X**2 + Y**2), dtype=np.float32)
    for i in range(num_movements+1):
        mask = masks[i]
        r_masked = r * mask
        weights[i] = np.sum(r_masked) / np.sum(r)
    weights = weights / np.sum(weights)
    return weights


def sampleMovements(lambda_large, lambda_small, max_large, max_small):
    """Sample number of large and small movements from Poisson distributions."""
    num_movements_large = min(np.random.poisson(lambda_large, 1)[0], max_large)
    num_movements_small = min(np.random.poisson(lambda_small, 1)[0], max_small)
    # If no movements, generate 1 large movement
    if num_movements_large == 0 and num_movements_small == 0:
        num_movements_large = 1
        num_movements_small = 0
    return num_movements_large, num_movements_small


def randomise3D(image_3d, cfg):
    """Randomise parameters for 3D motion."""

    if cfg['debug']:
        print('Randomising 3D...')

    # Randomise number of movements
    num_movements_large, num_movements_small = sampleMovements(cfg['lambda_large'],
                                                               cfg['lambda_small'],
                                                               cfg['max_large'],
                                                               cfg['max_small'])
    num_movements = num_movements_large + num_movements_small

    # Randomise angles
    angles_large = cfg['angles_stddev_large'] * np.random.randn(3, num_movements_large) * np.pi/180.
    angles_small = cfg['angles_stddev_small'] * np.random.randn(3, num_movements_small) * np.pi/180.
    angles = np.concatenate((angles_large, angles_small), axis=1)

    # Randomise translations
    trans_large = cfg['trans_stddev_large'] * np.random.randn(3, num_movements_large)
    trans_small = cfg['trans_stddev_small'] * np.random.randn(3, num_movements_small)
    trans = np.concatenate((trans_large, trans_small), axis=1)

    # Randomise movement durations (only instantaneous movements supported currently)
    durations = np.ones(num_movements, dtype=int)

    # Randomise times of movements
    assert cfg['max_kspace'] > cfg['min_kspace'], "max_kspace must be greater than min_kspace"
    n = 10000.
    times = np.sort(np.random.choice(range(int(cfg['min_kspace']*n),int(cfg['max_kspace']*n)), num_movements, replace=False)/n)

    # Shuffle movements
    inds = np.random.choice(num_movements, num_movements, replace=False)
    angles = angles[:,inds]
    trans = trans[:,inds]

    params = {'num_movements': num_movements,
              'times': times,
              'angles': angles,
              'trans': trans,
              'durations': durations}
    return params


def randomise2D(image_2d, cfg):
    """Randomise parameters for 2D motion."""

    if cfg['debug']:
        print('Randomising 2D...')

    # Randomise number of movements
    num_movements_large, num_movements_small = sampleMovements(cfg['lambda_large'],
                                                               cfg['lambda_small'],
                                                               cfg['max_large'],
                                                               cfg['max_small'])
    num_movements = num_movements_large + num_movements_small

    # Randomise angles
    angles_large = cfg['angles_stddev_large'] * np.random.randn(1, num_movements_large) * np.pi/180.
    angles_small = cfg['angles_stddev_small'] * np.random.randn(1, num_movements_small) * np.pi/180.
    angles = np.concatenate((angles_large, angles_small), axis=1)

    # Randomise translations
    trans_large = cfg['trans_stddev_large'] * np.random.randn(2, num_movements_large)
    trans_small = cfg['trans_stddev_small'] * np.random.randn(2, num_movements_small)
    trans = np.concatenate((trans_large, trans_small), axis=1)

    # Randomise movement durations (only instantaneous movements supported currently)
    durations = np.ones(num_movements, dtype=int)

    # Randomise times of movements
    assert cfg['max_kspace'] > cfg['min_kspace'], "max_kspace must be greater than min_kspace"
    if cfg['trajectory']=='cartesian':
        rows = image_2d.shape[0]
        min_rows = int(rows*cfg['min_kspace'])
        max_rows = int(rows*cfg['max_kspace'])
        times = np.sort(np.random.choice(np.arange(min_rows, max_rows), num_movements, replace=False).astype(int))
    elif cfg['trajectory'=='spiral']:
        #TO DO
        sys.exit('Not yet implemented')
        #r = int(min(image_2d.shape[:2])/2)
        #min_r = int(r*20.0/100)
        #times = np.sort(np.random.choice(np.arange(min_r,r), num_movements, replace=False).astype(int))
    elif cfg['trajectory'=='radial']:
        # TO DO
        sys.exit('Not yet implemented')

    # Shuffle movements
    inds = np.random.choice(num_movements, num_movements, replace=False)
    angles = angles[:,inds]
    trans = trans[:,inds]

    params = {'num_movements': num_movements,
              'times': times,
              'angles': angles,
              'trans': trans,
              'durations': durations}
    return params


def rand_motion_3d(image_3d, cfg=None):
    """Generate random 3D motion artefacts."""

    # Get config
    if cfg is None:
        cfg = get_default_config()

    # Randomise params
    params = randomise3D(image_3d, cfg)
    if cfg['debug']:
        print('Movements:\n', params['num_movements'])
        print('Times:\n', params['times'])
        print('Angles:\n', params['angles'] * 180./np.pi)
        print('Translations:\n', params['trans'])

    # Pad image
    image_3d = np.pad(image_3d, cfg['pad_width'], mode='edge')
    rows, cols, depth = image_3d.shape

    # Normalise image
    image_3d = normalise_image(image_3d)

    # Get 3D rotation matrices
    rotations = getRotations(params['angles'], mode='3D')

    # Create kspace masks
    masks = getMasks3D(image_3d, params['times'], params['num_movements'])

    # Compute weights (num_movements + 1 for identity transform)
    weights = computeWeights3D(masks, image_3d, params['times'], params['num_movements'])

    # Init
    F_composite = np.zeros_like(image_3d).astype(np.complex64)
    combinedAffines = []
    demeanedAffines = []
    Aprev = np.eye(4)
    Aavg = np.zeros((4,4))

    # Combine transforms and compute 'average' transform
    for i in range(params['num_movements']):
        R = rotations[i]
        t = params['trans'][:,i][:,None]
        A = getAffineMatrixITK(R, t)
        combinedA = expm( logm(A) + logm(Aprev) )
        combinedAffines.append(combinedA)
        Aavg = Aavg + weights[i+1]*logm(combinedA)
        Aprev = combinedA
    Aavg = expm(Aavg)
    Ainv = inv(Aavg)
    Ainv[3,:] = [0.,0.,0.,1.]

    # De-mean affine transforms
    for i in range(params['num_movements']):
        demeanedA = expm( logm(Ainv) + logm(combinedAffines[i]) )
        demeanedAffines.append(demeanedA)

    # De-mean inital image
    image_3d_transformed = affine3DTranformITK(image_3d, Ainv)

    # FFT of de-meaned image
    F_transformed = fft(image_3d_transformed)
    del image_3d_transformed
    gc.collect()

    # Apply Ainv mask before start of first movement
    mask = masks[0]
    F_composite[np.unravel_index(mask, F_composite.shape, 'F')] = F_transformed[np.unravel_index(mask, F_transformed.shape, 'F')]
    del F_transformed
    gc.collect()

    # Loop over movements
    for i in range(params['num_movements']):
        print(i+1, ' of ', params['num_movements'])
        duration = params['durations'][i]
        mask = masks[i+1]

        # Get start and end affine transforms
        if i == 0:
            Astart = np.copy(Ainv)
        else:
            Astart = demeanedAffines[i-1]
        Aend = demeanedAffines[i]

        for j in range(int(duration)):
            # Interpolate start and end transforms
            w = (j+1)/float(duration)
            Aj = expm( (1-w)*logm(Astart) + w*logm(Aend) )
            Aj[3,:] = [0,0,0,1]

            # Transform image
            image_3d_transformed = affine3DTranformITK(image_3d, Aj)

            # Fourier transform
            F_transformed = fft(image_3d_transformed)
            del image_3d_transformed
            gc.collect()

            # Apply mask at step j
            mj = mask[j]
            F_composite[np.unravel_index(mj, F_composite.shape, 'F')] = F_transformed[np.unravel_index(mj, F_transformed.shape, 'F')]

        # Apply end of mask
        mask_end = mask[duration:,]
        F_composite[np.unravel_index(mask_end, F_composite.shape, 'F')] = F_transformed[np.unravel_index(mask_end, F_transformed.shape, 'F')]
        del F_transformed
        gc.collect()

    # Inverse FFT
    image_3d = np.abs(ifft(F_composite), dtype=np.float32)
    del F_composite
    gc.collect()

    # Undo image padding
    image_3d = image_3d[cfg['pad_width']:rows-cfg['pad_width'],
                        cfg['pad_width']:cols-cfg['pad_width'],
                        cfg['pad_width']:depth-cfg['pad_width']]

    return image_3d


def rand_motion_2d(image_2d, cfg=None):
    """Generate random 2D motion artefacts."""

    # Get config
    if cfg is None:
        cfg = get_default_config()

    # Randomise params
    params = randomise2D(image_2d, cfg)
    if cfg['debug']:
        print('Movements:\n', params['num_movements'])
        print('Times:\n', params['times'] / image_2d.shape[0])
        print('Angles:\n', params['angles'] * 180./np.pi)
        print('Translations:\n', params['trans'])

    # Pad image
    pad_width = 20
    image_2d = np.pad(image_2d, cfg['pad_width'], mode='edge')
    rows, cols = image_2d.shape

    # Normalise image
    image_2d = normalise_image(image_2d)

    # Get 2D rotation matrices
    rotations = getRotations(params['angles'], mode='2D')

    # Create kspace masks
    masks = getMasks2D(image_2d, params['times'], params['num_movements'])

    # Compute weights (num_movements + 1 for identity transform)
    weights = computeWeights2D(masks, image_2d, params['times'], params['num_movements'])

    # Init
    F_composite = np.zeros_like(image_2d).astype(np.complex64)
    combinedAffines = []
    demeanedAffines = []
    Aprev = np.eye(3)
    Aavg = np.zeros((3,3))
    if cfg['debug']:
        imgs, ffts, ffts_masked = [], [], []

    # Combine transforms and compute 'average' transform
    for i in range(params['num_movements']):
        R = rotations[i]
        t = params['trans'][:,i][:,None]
        A = getAffine2DMatrixITK(R, t)
        combinedA = expm( logm(A) + logm(Aprev) )
        combinedAffines.append(combinedA)
        Aavg = Aavg + weights[i+1]*logm(combinedA)
        Aprev = combinedA
    Aavg = expm(Aavg)
    Ainv = inv(Aavg)
    Ainv[2,:] = [0.,0.,1.]

    # De-mean affine transforms
    for i in range(params['num_movements']):
        demeanedA = expm( logm(Ainv) + logm(combinedAffines[i]) )
        demeanedAffines.append(demeanedA)

    # De-mean inital image
    image_2d_transformed = affine2DTranformITK(image_2d, Ainv)

    # FFT of de-meaned image
    F_transformed = fft2D(image_2d_transformed)

    # Apply Ainv mask before start of first movement
    mask = masks[0]
    F_composite = F_transformed * mask

    if cfg['debug']:
        imgs.append(image_2d_transformed)
        ffts.append(F_transformed)
        ffts_masked.append(F_transformed * mask)

    del image_2d_transformed, F_transformed
    gc.collect()

    # Loop over movements
    for i in range(params['num_movements']):
        if cfg['debug']:
            print(i+1, ' of ', params['num_movements'])

        # Get mask
        mask = masks[i+1]

        # Get start and end affine transforms
        if i == 0:
            Astart = np.copy(Ainv)
        else:
            Astart = demeanedAffines[i-1]
        Aend = demeanedAffines[i]

        # Transform image
        image_2d_transformed = affine2DTranformITK(image_2d, Aend)

        # FFT transform
        F_transformed = fft2D(image_2d_transformed)

        # Apply mask and composite FFT
        F_composite = F_composite + F_transformed * mask

        if cfg['debug']:
            imgs.append(image_2d_transformed)
            ffts.append(F_transformed)
            ffts_masked.append(F_transformed * mask)

        # Clean up
        del image_2d_transformed, F_transformed
        gc.collect()

    # Inverse FFT
    image_2d = np.abs(ifft2D(F_composite), dtype=np.float32)

    # Undo image padding
    image_2d = image_2d[cfg['pad_width']:rows-cfg['pad_width'], 
                        cfg['pad_width']:cols-cfg['pad_width']]

    if cfg['debug']:
        fig, axs = plt.subplots(5,params['num_movements']+1)
        for i in range(params['num_movements']+1):
            axs[0,i].imshow(masks[i], vmin=0, vmax=1)
            axs[1,i].imshow(imgs[i], vmin=0, vmax=1)
            axs[2,i].imshow(np.log(np.abs(ffts[i])+1))
            axs[3,i].imshow(np.log(np.abs(ffts_masked[i])+1))
            axs[4,i].imshow(np.abs(ifft2D(ffts_masked[i]), dtype=np.float32))
            axs[0,i].set_title('t: %d' % i)
        fig = plt.figure()
        plt.imshow(np.log(np.abs(F_composite)+1))
        plt.title('Composite k-space')
        plt.show()

    # Clean up
    del F_composite
    gc.collect()

    return image_2d
