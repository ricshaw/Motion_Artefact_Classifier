import os
import sys
import numpy as np
import pandas as pd
import cv2
import nibabel as nib
import random
import time
from PIL import Image
from PIL.Image import fromarray
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, GroupKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import roc_curve, auc, roc_auc_score, average_precision_score

import torch
from torch.utils.data import TensorDataset, DataLoader,Dataset
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.data import NiftiDataset
from monai.transforms import LoadNifti, Randomizable, apply_transform
from monai.transforms import AddChannel, Compose, RandRotate90, Resize, ScaleIntensity, ToTensor, RandAffine, RandFlip, RandAdjustContrast, RandScaleIntensity
from monai.utils import get_seed

import albumentations as A

ROOT = '/nfs/home/richard/Motion_Artefact_Classifier'

sys.path.append(os.path.join(ROOT, 'MRI_motion_model'))
from rand_motion import rand_motion_3d, rand_motion_2d

sys.path.append(os.path.join(ROOT, 'over9000'))
from rangerlars import RangerLars


# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', device)


# Set seed
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    print('Seeded!')
set_seed(0)


# Set motion augmentation config
cfg = {'lambda_large': 1,
       'lambda_small': 2,
       'max_large': 2,
       'max_small': 5,
       'angles_stddev_large': 5, #10.,
       'angles_stddev_small': 2, #3.,
       'trans_stddev_large': 5, #10.,
       'trans_stddev_small': 2, #2.,
       'min_kspace': 0.1, #0.25,
       'max_kspace': 0.9, #0.75,
       'pad_width': 20,
       'trajectory':'cartesian',
       'debug':False}


# Set parameters
DATA_DIR = '/nfs/project/richard/ADNI_resampled/ADNI_QC1'
MODE = 'train'
#MODE = 'inference'
batch_size = 16
lr = 1e-3
dropout = 0.2
label_smoothing = 0.1
epochs = 100
artefact_prob = 0.5
out_dim = 1
num_workers = 4
SAVE = True
SAVE_INTERVAL = 2
SAVE_NAME = 'model_classifier3d_bs%d_lr%.03f_dp%.1f' % (batch_size, lr, dropout)
print('Model:', SAVE_NAME)

SAVE_PATH = os.path.join(ROOT, SAVE_NAME)
if SAVE and MODE == 'train':
    os.makedirs(SAVE_PATH, exist_ok=True)
    log_name = os.path.join(SAVE_PATH, 'run')
    writer = SummaryWriter(log_dir=log_name)


def normalise_image(image):
    """Normalise image 0 to 1"""
    if (image.max() - image.min()) < 1e-5:
        return image - image.min() + 1e-5
    else:
        return (image - image.min()) / (image.max() - image.min())


def load_nii(filename):
    """Load nifty volumne"""
    img = nib.load(filename)
    img = np.asanyarray(img.dataobj).astype(np.float32)
    return normalise_image(img)


def load_image(filename, A_transform=None):
    """Load 2D image"""
    img = cv2.imread(filename)[...,1]
    img = np.expand_dims(img, axis=-1)

    if A_transform is not None:
        img = A_transform(image=img)['image']

    img = img.astype(np.float32) / 255.0
    return img.transpose(2,0,1)


def sample_slices(img, num):
    """Sample 2D slices from volume"""
    slices = img[..., np.random.choice(img.shape[-1], num, replace=False)]
    return slices


class ImageDataset(Dataset):
    """Image data loader"""
    def __init__(self, data_dir, filelist, mode, transform=None):
        self.data_dir = data_dir
        self.filelist = filelist
        self.mode = mode
        self.transform = transform
        self.eps = label_smoothing

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        # Load volume
        name = self.filelist[index]
        filename = os.path.join(self.data_dir, name)
        img = load_nii(filename)

        # Normalise
        img = normalise_image(img)

        if self.mode == 'test':
            return img, filename

        # Apply Monai augmentatiopn
        if self.transform is not None:
            seed = np.random.randint(np.iinfo(np.uint32).max + 1, dtype="uint32")
            self.transform.set_random_state(seed=seed)
            img = apply_transform(self.transform, img)

        label = []
        # Apply motion artefact
        if np.random.random() < artefact_prob:
            img = rand_motion_3d(img, cfg=cfg)
            label.append(1.0-self.eps) # label=1
        else:
            label.append(self.eps) # label=0
        label = np.array(label).astype(np.float64)
        img = img.astype(np.float32)
        return img, label


# Set Monai augmentations
train_transform = Compose([
                           RandScaleIntensity(prob=0.5, factors=0.2),
                           RandAdjustContrast(prob=0.5, gamma=(0.8, 1.2)),
                           RandAffine(
                                      prob=0.5,
                                      translate_range=(10, 10, 10),
                                      rotate_range=(np.pi*2, np.pi*2, np.pi*2),
                                      scale_range=(0.15, 0.15, 0.15),
                                      padding_mode='border',
                                      as_tensor_output=False),
                          ])


# BCE loss
bce = nn.BCEWithLogitsLoss()
def criterion(logits, target):
    loss = bce(logits, target)
    return loss


# Label smoothing loss
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


# Define model
def get_model():
    model = monai.networks.nets.densenet.densenet121(spatial_dims=3, in_channels=1, out_channels=1, dropout_prob=dropout).to(device)
    model = nn.DataParallel(model)
    return model

def load_model(model, MODEL_PATH):
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
        print('Loaded:', MODEL_PATH)
    else:
        print('No model!')
        sys.exit(0)


# Training epoch
def train_epoch(model, loader, optimizer):
    model.train()
    train_loss = []
    epoch_loss = 0
    y_pred, y_true = [], []

    for step, (data, target) in enumerate(loader):
        data, target = data.to(device).float(), target.to(device).float()
        data = data.unsqueeze(1)
        target = target.view(-1,1)
        optimizer.zero_grad()

        logits = model(data)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()

        loss_np = loss.detach().cpu().numpy()
        epoch_loss += loss_np
        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        print('loss: %.5f, smooth loss: %.5f' % (loss_np, smooth_loss))

        y_prob = torch.sigmoid(logits)
        y_pred += y_prob.detach().cpu().numpy().tolist()
        y_true += target.detach().cpu().numpy().tolist()

    epoch_loss /= (step+1)
    y_true = np.round(np.array(y_true)).astype(int)
    y_pred = np.array(y_pred)
    acc = balanced_accuracy_score(y_true, (y_pred>0.5).astype(int))
    try:
        auc = roc_auc_score(y_true, y_pred)
    except:
        auc = 0.0

    target_imgs = []
    for t in target:
        target_imgs.append( t.item() * torch.ones((32,32)) )
    target_imgs = torch.stack(target_imgs)
    target_imgs = target_imgs.unsqueeze(1)

    id = target.argmax().item()
    return epoch_loss, acc, auc, data[id,...], target_imgs


# Validation epoch
def val_epoch(model, loader):
    model.eval()
    epoch_loss = 0
    y_pred, y_true = [], []

    with torch.no_grad():
        for step, (data, target) in enumerate(loader):
            data, target = data.to(device).float(), target.to(device).float()
            data = data.unsqueeze(1)
            target = target.view(-1,1)

            logits = model(data)
            loss = criterion(logits, target)

            loss_np = loss.detach().cpu().numpy()
            epoch_loss += loss_np
            y_prob = torch.sigmoid(logits)
            y_pred += y_prob.detach().cpu().numpy().tolist()
            y_true += target.detach().cpu().numpy().tolist()

        epoch_loss /= (step+1)
        y_true = np.round(np.array(y_true)).astype(int)
        y_pred = np.array(y_pred)
        acc = balanced_accuracy_score(y_true, (y_pred>0.5).astype(int))
        try:
            auc = roc_auc_score(y_true, y_pred)
        except:
            auc = 0.0

    id = target.argmax().item()
    return epoch_loss, acc, auc, data[id,...]


# Set random worker seed
def wif(worker_id):
    np.random.seed()


def train():
    # Train/Val/Test split
    all_files = sorted(os.listdir(DATA_DIR))
    train_list, test_list, = train_test_split(all_files, test_size=0.2, random_state=1)
    train_list, val_list = train_test_split(train_list, test_size=0.25, random_state=1)
    print(len(train_list), len(val_list), len(test_list))

    # Datasets
    train_dataset = ImageDataset(DATA_DIR, train_list, mode='train', transform=train_transform)
    valid_dataset = ImageDataset(DATA_DIR, val_list, mode='valid', transform=None)

    # Sampler
    sampler = torch.utils.data.sampler.RandomSampler(train_dataset, replacement=False)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler, pin_memory=True, worker_init_fn=wif)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)
    print('Train:', len(train_dataset), 'Valid:', len(valid_dataset))

    # Model
    model = get_model()
    #optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = RangerLars(model.parameters(), lr=lr)

    # Epoch loop
    for epoch in range(1, epochs+1):
        print(time.ctime(), 'Epoch:', epoch)

        # Train epoch
        train_loss, train_acc, train_auc, train_images, train_labels = train_epoch(model, train_loader, optimizer)
        print('train loss:', train_loss, 'acc:', train_acc, 'auc:', train_auc)
        label_grid = torchvision.utils.make_grid(train_labels, nrow=8, normalize=False, scale_each=False)
        writer.add_images('train/images', train_images, epoch, dataformats='CNHW')
        writer.add_image('train/labels', label_grid, epoch)
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Acc/train', train_acc, epoch)
        writer.add_scalar('AUC/train', train_auc, epoch)

        # Val epoch
        val_loss, val_acc, val_auc, val_images = val_epoch(model, valid_loader)
        print('val loss:', val_loss, 'acc:', val_acc, 'auc:', val_auc)
        writer.add_images('val/images', val_images, epoch, dataformats='CNHW')
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Acc/val', val_acc, epoch)
        writer.add_scalar('AUC/val', val_auc, epoch)

        # Save
        if epoch % SAVE_INTERVAL == 0:
            MODEL_PATH = os.path.join(SAVE_PATH, f'{SAVE_NAME}_epoch{epoch}.pth')
            print('Saving', MODEL_PATH)
            torch.save(model.state_dict(), MODEL_PATH)


def test():
    # Train/Val/Test split
    all_files = sorted(os.listdir(DATA_DIR))
    train_list, test_list, = train_test_split(all_files, test_size=0.2, random_state=1)
    train_list, val_list = train_test_split(train_list, test_size=0.25, random_state=1)
    print(len(train_list), len(val_list), len(test_list))

    # Test Dataset
    test_dataset = ImageDataset(DATA_DIR, test_list, mode='valid', transform=None)

    # Test Dataloader
    test_loader = DataLoader(test_dataset, batch_size=4, num_workers=num_workers, shuffle=False, pin_memory=True)
    print('Test:', len(test_dataset))

    # Model
    model = get_model()
    MODEL_PATH = os.path.join(ROOT, 'model_classifier3d_bs4_lr0.001_dp0.2/model_classifier3d_bs4_lr0.001_dp0.2_4.pth')
    load_model(model, MODEL_PATH)

    # Test
    test_loss, test_acc, test_auc, test_images = val_epoch(model, test_loader)
    print('test loss:', test_loss, 'acc:', test_acc, 'auc:', test_auc)
    return


# Inference
def inference():
    """Run inference on a single input volume"""
    # Load image
    img = load_nii(os.path.join(ROOT, 'test.nii.gz'))
    img = normalise_image(img)
    #img = rand_motion_3d(img, cfg=cfg)

    data = torch.tensor(img).to(device).float()
    data = data.unsqueeze(0).unsqueeze(0)
    print(data.shape)

    # Load model
    model = get_model()
    MODEL_PATH = os.path.join(ROOT, 'model_classifier3d_bs4_lr0.001_dp0.2/model_classifier3d_bs4_lr0.001_dp0.2_epoch10.pth')
    load_model(model, MODEL_PATH)

    # Inference
    print('Running inference...')
    model.eval()
    with torch.no_grad():
        logits = model(data)
        y_prob = torch.sigmoid(logits).item()
        print(y_prob)
    return


def inference_folder():
    """Run inference on a whole folder"""
    # Dataset
    DATA_DIR = '/nfs/project/richard/ADNI3_resampled/Motion/Fail'
    test_list = sorted(os.listdir(DATA_DIR))
    test_dataset = ImageDataset(DATA_DIR, test_list, mode='test', transform=None)
    y_true = np.ones((len(test_list)))

    # Dataloader
    test_loader = DataLoader(test_dataset, batch_size=4, num_workers=num_workers, shuffle=False, pin_memory=True)

    # Load model
    model = get_model()
    MODEL_PATH = os.path.join(ROOT, 'model_classifier3d_bs4_lr0.001_dp0.2/model_classifier3d_bs4_lr0.001_dp0.2_epoch10.pth')
    load_model(model, MODEL_PATH)

    # Inference
    print('Running inference...')
    model.eval()
    y_pred = []
    with torch.no_grad():
        for step, (data, name) in enumerate(test_loader):
            data = data.to(device).float()
            data = data.unsqueeze(1)
            logits = model(data)
            y_prob = torch.sigmoid(logits)
            y_pred += y_prob.detach().cpu().numpy().tolist()
            for i in range(len(y_prob)):
                print(name[i], y_prob[i].item())

    y_pred = np.array(y_pred)
    thresh = np.arange(0.1,1.0,0.1)
    for i in range(len(thresh)):
        th = thresh[i]
        acc = balanced_accuracy_score(y_true, (y_pred>th).astype(int))
        print('thresh: %.1f acc: %.4f' % (th, acc))
    auc = roc_auc_score(y_true, y_pred)
    print('auc: %.4f' % auc)
    return



if MODE == 'train':
    train()

if MODE == 'test':
    test()

if MODE == 'inference':
    inference()
    #inference_folder()
