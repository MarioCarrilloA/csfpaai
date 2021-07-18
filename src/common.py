import json
import io
import matplotlib.pyplot as plt
import numpy as np
import os.path
import PIL
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from datadings.sets import ImageClassificationData
from datadings.writer import FileWriter
from datadings import reader as ddreader
from matplotlib.pyplot import imshow
from multiprocessing import Lock
from PIL import Image
from sacred import Experiment
from sacred.observers import file_storage
from sklearn.model_selection import train_test_split
from torch import topk
from torch.autograd import Variable
from torch.utils.data import random_split, Dataset, Subset
from torchvision import models, datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm
from torchcam.cams import CAM

# Check for GPU/CPU to allocate tensor
device = None
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# Classes labels CIFAR-10
classes = ('plane',
           'auto',
           'bird',
           'cat',
           'deer',
           'dog',
           'frog',
           'horse',
           'ship',
           'truck')

# CIFAR1-10 parameters
input_size = 32
num_classes = 10
out_filename = 'results.json'

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip()
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
])

