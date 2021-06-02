#import matplotlib.pyplot as plt
import numpy as np
import random
import skimage.transform
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision

import os.path
import random
import json
#from keras.optimizers import SGD
#from matplotlib.pyplot import imshow
#from PIL import Image
from torch.autograd import Variable
from torch import topk
from torchvision import models, datasets, transforms
from sklearn.model_selection import train_test_split


###############################################################################
#    GLOBAL VARIABLES
###############################################################################
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
###############################################################################

###############################################################################
#  CLASSES
###############################################################################

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet = models.resnet18(pretrained=False, num_classes=10)

        self.resnet.conv1 = torch.nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.resnet.maxpool = torch.nn.Identity()

    def forward(self, x):
        x = self.resnet(x)
        x = F.log_softmax(x, dim=1)

        return x

###############################################################################



def foo():
    print("module loaded correctly!")



def build_model(train_loader, test_loader, e, model_file="PAAI21_CIFAR10_model.pt"):
    # Hyperparameters
    epochs=e
    lr=0.1

    model = Model()
    model = model.cuda()

    milestones = [25, 40]

    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=milestones, gamma=0.1)

    train_loss = []
    test_loss = []
    print("Start train/test resnet18!")
    for epoch in range(1, epochs + 1):
        avg_loss = train_model(model, train_loader, optimizer, epoch)
        loss, pct_correct, pct_classes = test_model(model, test_loader)
        output = format_model_output(epoch, avg_loss, loss, pct_correct, pct_classes)
        train_loss.append(avg_loss)
        test_loss.append(loss.item())
        print(output)

        scheduler.step()

    torch.save(model.state_dict(), model_file)

    return model, pct_correct, pct_classes, train_loss, test_loss


def train_model(model, train_loader, optimizer, epoch, verbose=False):
    model.train()
    total_loss = []

    for data, target in train_loader:
        data = data.cuda()
        target = target.cuda()

        optimizer.zero_grad()
        prediction = model(data)
        loss = F.nll_loss(prediction, target)

        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())

        avg_loss = sum(total_loss) / len(total_loss)
        if (verbose):
            print("Training set: Epoch: {} Average Loss: {:.2f}".format(epoch, avg_loss))

    return avg_loss


def _get_classes_percentage(targets, predictions):
    num_classes = len(classes)
    num_samples = len(targets)

    # Init class lists
    correct_per_class = [0] * num_classes
    total_per_class = [0] * num_classes
    percentage_per_class = [0] * num_classes
    for i in range(num_samples):
        class_id = targets[i]
        if targets[i] == predictions[i]:
            correct_per_class[class_id] += 1
        total_per_class[class_id] += 1

    for i in range(num_classes):
        percentage_per_class[i] = 100 * (correct_per_class[i] / total_per_class[i])

    return percentage_per_class


def test_model(model, test_loader, verbose=False):
    model.eval()
    loss = 0
    correct = 0

    for data, target in test_loader:
        with torch.no_grad():
            data = data.cuda()
            target = target.cuda()

            prediction = model(data)
            loss += F.nll_loss(prediction, target, reduction="sum")

            prediction = prediction.max(1)[1]
            correct += prediction.eq(target.view_as(prediction)).sum().item()

            loss /= len(test_loader.dataset)
            percentage_correct = 100.0 * correct / len(test_loader.dataset)
            percentage_classes = _get_classes_percentage(target, prediction)

            if (verbose):
                print("Testing set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
                    loss, correct, len(test_loader.dataset), percentage_correct))

    return loss, percentage_correct, percentage_classes

def collect_results(epochs, accuracy, train_loss, test_loss, classes_pcts, out):
    file_name = out
    data_file = []
    result_id = 0
    if os.path.isfile(file_name) == True:
        with open(file_name, 'r') as json_file:
            data_file = json.load(json_file)
            result_id = len(data_file) + 1
            json_file.close()
    else:
        result_id = 1

    res = {
        "result_id" : result_id,
        "epochs" : epochs,
        "accuracy" : accuracy,
        "train_loss" : train_loss,
        "test_loss" : test_loss,
        "classes_pcts": classes_pcts
    }
    data_file.append(res)
    with open(file_name, 'w') as json_file:
        json.dump(data_file, json_file, ensure_ascii=True, indent=4)
        json_file.close()

def format_model_output(e, avg_loss, tloss, pct_correct, pct_classes):
    output  = "Epoch:{: <2} ".format(e)
    output += "TrainLoss:{: <4.2f} ".format(avg_loss)
    output += "TestLoss:{: <4.2f} ".format(tloss)
    output += "Accuracy:{: <5.2f}% ".format(pct_correct)

    num_classes = len(classes)
    for i in range(num_classes):
        output += "{}:{: <5.2f}% ".format(classes[i], pct_classes[i])

    return output
