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


class LayerFeatures():
    features=None
    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output

    def remove(self): self.hook.remove()


def compute_CAM(feature_conv, class_weights):
    _, nc, h, w = feature_conv.shape
    feature_conv = feature_conv.reshape(nc, h*w)
    CAM = class_weights.matmul(feature_conv)
    CAM = CAM.reshape(h, w)
    CAM = CAM - CAM.min()
    CAM = CAM / CAM.max()

    return CAM

def get_one_random_sample(test_dataset):
    num_total_imgs = len(test_dataset.data)
    random_index = random.randint(1, num_total_imgs)
    img = test_dataset.data[random_index]
    label = test_dataset.targets[random_index]

    return img, label


def get_heatmaps(tensor, model):
    prediction_var = Variable((tensor.unsqueeze(0)).cuda(), requires_grad=True)
    final_layer = model._modules.get("resnet").layer4[-1]
    activated_features = LayerFeatures(final_layer)

    prediction = model(prediction_var)
    pred_probabilities = F.softmax(prediction, dim=1).data.squeeze()
    activated_features.remove()

    # Indentify the predicted class
    value, index = topk(pred_probabilities, 1)

    # Get information from identified class
    weight_softmax_params = list(model._modules.get('resnet').fc.parameters())
    weight_softmax = weight_softmax_params[0]
    class_id = topk(pred_probabilities,1)[1].int()
    class_index = [class_id.item()]
    class_weights = weight_softmax[class_index]
    cam_img = compute_CAM(activated_features.features, class_weights)

    # As we can see, our CAM size does not match with the our
    # image. We need to resize our map and interpolate the values
    # according to our image
    ucam_img = cam_img.unsqueeze(dim=0)
    ucam_img = ucam_img[None, :, :, :]
    heat_map = F.interpolate(ucam_img, size=(32, 32), mode='bilinear')
    heat_map = heat_map.squeeze(0)
    heat_map = heat_map.squeeze(0)

    return cam_img, heat_map, index, value


def crop_preprocess(x, model):
    # The 5% of the highest values represent a one of the most
    # important values for classification. These values will be
    # for experiments modified.
    device = 'cuda'
    x = x.to(device)
    cam_img, heat_map, index, value = get_heatmaps(x, model)
    percentile = 95
    h, w = heat_map.shape
    feature_thld = torch.quantile(heat_map, percentile * 0.01)
    heat_mask = heat_map.clone()
    bk_mask = heat_map.clone()
    heat_mask = torch.where(heat_mask >= feature_thld, 0.0, 1.0)
    bk_mask =  torch.where(bk_mask >= feature_thld, 1.0, 0.0) # Replace value
    x = torch.mul(x, heat_mask)
    x = x + bk_mask

    return x.float()

def show_sample_images(train_loader):
    num_imgs_toshow= 10
    data_iter = iter(train_loader)
    images, labels = data_iter.next()
    # convert images to numpy for display
    images = images.cpu().numpy()

    # plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(15, 5))
    for i in np.arange(num_imgs_toshow):
        ax = fig.add_subplot(2, num_imgs_toshow/2, i + 1, xticks=[], yticks=[])
        plt.imshow(np.transpose(images[i], (1, 2, 0)))

