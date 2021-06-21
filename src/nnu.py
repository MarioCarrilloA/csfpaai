
import csv
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path
import pandas as pd
import random
import shutil
import skimage.transform
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from matplotlib.pyplot import imshow
from PIL import Image
from sacred import Experiment
from sacred.observers import file_storage
from skimage import io
from sklearn.model_selection import train_test_split
from torch import topk
from torch.autograd import Variable
from torch.utils.data import random_split
from torch.utils.data import Dataset
from torch.utils.data import Subset
from torchvision import models, datasets, transforms
from torchvision.utils import save_image

from datadings.writer import FileWriter


import os
import io
import sys
import numpy as np
from multiprocessing import Lock

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import datasets

# Use when decoding images from Datadings directly to PIL objects
import PIL

# Use when decoding images from Datadings directly to PIL objects (faster than PIL)
import simplejpeg as sjpg


# Check for Datadings support
import datadings
from datadings import reader as ddreader



#####################################################################################################
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
epochs=50
out_filename = 'results.json'

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip()
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
])

#####################################################################################################

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

class LayerFeatures():
    features=None
    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output

    def remove(self): self.hook.remove()


class croppedCIFAR10(Dataset):
    def __init__(self,
                 root,
                 train=True,
                 transform=None):
        super().__init__()
        split_to_load = 'train' if train else 'test'
        feat_file = os.path.join(root, '{}.msgpack'.format(split_to_load))
        self._reader = ddreader.MsgpackReader(feat_file)
        self._iter = self._reader.rawiter(False)
        self.size = len(self._reader)
        self.transform = transform
        self.root = root
        self.unpack_lock = Lock()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        with self.unpack_lock:
            self._reader.seek(idx)
            raw_sample = ddreader.msgpack.unpackb(next(self._iter))

        x = raw_sample['image']
        x = np.transpose(x, (1, 2, 0))
        if self.transform:
            image = self.transform(x)

        return (image, raw_sample['label'])


def collect_results(res, out):
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

    data_file.append(res)
    with open(file_name, 'w') as json_file:
        json.dump(data_file, json_file, ensure_ascii=True, indent=4)
        json_file.close()


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


def compute_CAM(feature_conv, class_weights):
    _, nc, h, w = feature_conv.shape
    feature_conv = feature_conv.reshape(nc, h*w)
    CAM = class_weights.matmul(feature_conv)
    CAM = CAM.reshape(h, w)
    CAM = CAM - CAM.min()
    CAM = CAM / CAM.max()

    return CAM


def get_one_random_sample(test_dataset):
    if isinstance(test_dataset, croppedCIFAR10):
        num_total_imgs = len(test_dataset)
        random_index = random.randint(1, num_total_imgs - 1)
        imx = test_dataset[random_index][0]
        img = transforms.ToPILImage()(imx).convert("RGB")
        label = test_dataset[random_index][1]
    elif isinstance(test_dataset, Subset):
        num_total_imgs = len(test_dataset.dataset.data)
        random_index = random.randint(1, num_total_imgs)
        img = test_dataset.dataset.data[random_index]
        label = test_dataset.dataset.targets[random_index]
    else:
        num_total_imgs = len(test_dataset.data)
        random_index = random.randint(1, num_total_imgs)
        img = test_dataset.data[random_index]
        label = test_dataset.targets[random_index]

    return img, label


def get_classes_percentage(targets, predictions):
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
        if total_per_class[i] == 0:
            percentage_per_class[i] = 0
        else:
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
            percentage_classes = get_classes_percentage(target, prediction)

            if (verbose):
                print("Testing set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
                    loss, correct, len(test_loader.dataset), percentage_correct))

    return loss, percentage_correct, percentage_classes


def format_model_output(e, avg_loss, tloss, pct_correct, pct_classes):
    output  = "Epoch:{: <2} ".format(e)
    output += "TrainLoss:{: <4.2f} ".format(avg_loss)
    output += "TestLoss:{: <4.2f} ".format(tloss)
    output += "Accuracy:{: <5.2f}% ".format(pct_correct)

    num_classes = len(classes)
    for i in range(num_classes):
        output += "{}:{: <5.2f}% ".format(classes[i], pct_classes[i])

    return output

def build_model(train_loader,
        test_loader,
        epochs,
        lr,
        model_file="PAAI21_CIFAR10_model.pt"):

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

    metrics = {
            "accuracy" : pct_correct,
            "train_loss" : train_loss,
            "test_loss" : test_loss,
            "classes_pcts": pct_classes}

    return model, metrics

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
    # according to our image.
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


def save_sample(original, cam, heat_map, crop, acc, tgt, predt, out):
    vcam = cam.cpu().data.numpy()
    vheat_map = heat_map.cpu().data.numpy()
    res = "FAIL!"
    if tgt == predt:
        res = "CORRECT!"

    plt.clf()
    f, ax = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
    f.suptitle('ACCURACY: {}% TARGET: {}  PREDICTION: {} - {}'.format(acc, tgt, predt, res), fontsize=20)
    ax[0].imshow(original)
    ax[0].set_title("Original")
    ax[1].imshow(vcam, alpha=0.5, cmap='jet')
    ax[1].set_title("Class activation map")
    ax[2].imshow(original)
    ax[2].imshow(vheat_map, alpha=0.5, cmap='jet')
    ax[2].set_title("Heat map")
    ax[3].imshow(np.transpose(crop.cpu(), (1, 2, 0)))
    ax[3].set_title("Cropped image")
    plt.savefig(out, bbox_inches='tight')


def save_random_samples(model_base, num_samples, crop_transformation, test_dataset, prefix=1):
    for i in range(num_samples):
        # CAM
        image, label =  get_one_random_sample(test_dataset)
        image_tensor = test_transform(image)
        cam_img, heat_map, index, value = get_heatmaps(image_tensor, model_base)
        cropped_image = crop_transformation(image)
        prediction = classes[index.tolist()[0]]
        target = classes[label]
        acc = round(value.tolist()[0] * 100, 2)
        out = "../res/sample{}.{}.png".format(prefix, i)
        save_sample(image, cam_img, heat_map, cropped_image,
            acc, target, prediction, out)


def create_new_dataset(dset, new_data, csv_file, crop_transformation, train=True):
    row_list = []
    count = 0
    out_type = 'train'
    if train == False:
        out_type = 'test'
    i = 0
    outfile = new_data + out_type + '.msgpack'
    count = 0
    writer = FileWriter(outfile, len(dset), overwrite=True)

    # For croppedCIFAR10 customized dataset
    if isinstance(dset, croppedCIFAR10):
        for e in dset:
            image = e[0]
            label = e[1]
            cropped_image = crop_transformation(transforms.ToPILImage()(image).convert("RGB"))
            image_name = classes[label] + "_" + str(i) + ".png"
            image_path = new_data + image_name
            PIL_image = cropped_image.cpu().detach().numpy()
            sample = {"key": image_name, "image": PIL_image, "label": label}
            writer.write(sample)
            i+=1
            count += 1
            #if count == 100:
            #    break
        writer.close()

    else:
        for i in range(len(dset)):
            image = dset.dataset.data[i]
            label = dset.dataset.targets[i]
            cropped_image = crop_transformation(image)
            image_name = classes[label] + "_" + str(i) + ".png"
            image_path = new_data + image_name
            PIL_image = cropped_image.cpu().detach().numpy()
            sample = {"key": image_name, "image": PIL_image, "label": label}
            writer.write(sample)
            count += 1
            #if count == 100:
            #    break
        writer.close()


def save_dataset_samples(train_loader, out):
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
    plt.savefig(out, bbox_inches='tight')

