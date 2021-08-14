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
import torchcam

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
from torchcam.cams import GradCAM


def get_dataset_components(dataset_name):
    # No data augmentation for testing, ony tensor
    # ransformation.
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    if dataset_name == 'CIFAR10':
        # Classes labels CIFAR-10
        classes = (
                'plane',
                'auto',
                'bird',
                'cat',
                'deer',
                'dog',
                'frog',
                'horse',
                'ship',
                'truck'
        )

        # Data augmentation for CIFAR10
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()
        ])

        # Download and transform train/test CIFAR10
        train_full_dataset = datasets.CIFAR10(
                    "./data",
                    train=True,
                    transform=train_transform,
                    download=True
        )
        test_dataset = datasets.CIFAR10(
                    "./data",
                    train=False,
                    transform=test_transform,
                    download=True
        )

    elif dataset_name == 'STL10':
        # Classes labels STL-10
        classes = (
                'plane',
                'auto',
                'bird',
                'cat',
                'deer',
                'dog',
                'horse',
                'monkey',
                'ship',
                'truck'
        )

        # Data augmentation for CIFAR10
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(96, padding=4),
            transforms.RandomHorizontalFlip()
        ])

        # Download & transform STL-10 datasets
        train_full_dataset = datasets.STL10(
                    "./data",
                    split='train',
                    transform=train_transform,
                    download=True
        )
        test_dataset = datasets.STL10(
                    "./data",
                    split='test',
                    transform=test_transform,
                    download=True
        )

    else:
        return None, None, None, None, None, None

    # Split datasets in 90% for training set and 10% for Validation set.
    train_num_samples = int(len(train_full_dataset) * 0.9)
    val_num_samples = int(len(train_full_dataset) * 0.1)
    train_dataset, validation_dataset = random_split(
            train_full_dataset,
            [train_num_samples, val_num_samples]
    )

    return train_dataset, test_dataset, validation_dataset, train_transform, test_transform, classes


