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
from torchcam.cams import GradCAMpp
from torchcam.cams import SmoothGradCAMpp
from torchcam.cams import ScoreCAM
from torchcam.cams import SSCAM
from torchcam.cams import ISCAM


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


class croppedDataset(Dataset):
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

        # Decode image
        bytes_buf = io.BytesIO(raw_sample['image'])
        im = PIL.Image.open(bytes_buf)
        if im.mode != 'RGB':
            im = im.convert('RGB')

        if self.transform:
            im = self.transform(im)

        return im, raw_sample['label']


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
    if isinstance(test_dataset, croppedDataset):
        num_total_imgs = len(test_dataset)
        random_index = random.randint(1, num_total_imgs - 1)
        imx = test_dataset[random_index][0]
        img = transforms.ToPILImage()(imx).convert("RGB")
        label = test_dataset[random_index][1]
    elif isinstance(test_dataset, Subset):
        num_total_imgs = len(test_dataset.dataset.data)
        random_index = random.randint(1, round(num_total_imgs * 0.90))
        img, label = test_dataset[random_index]
        img = transforms.ToPILImage()(img).convert("RGB")
    else:
        num_total_imgs = len(test_dataset.data)
        random_index = random.randint(1, num_total_imgs)
        img = test_dataset.data[random_index]
        label = test_dataset.targets[random_index]

    return img, label


def get_classes_percentage(targets, predictions, classes):
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


def compute_pct_perclass(correct_per_class, total_per_class, classes):
    num_classes = len(classes)
    percentage_per_class = [0] * num_classes
    for i in range(num_classes):
        if total_per_class[i] == 0:
            percentage_per_class[i] = 0
        else:
            percentage_per_class[i] = 100 * (correct_per_class[i] / total_per_class[i])

    return percentage_per_class



def test_model(model, dataset_loader, classes, verbose=False):
    model.eval()
    loss = 0
    correct = 0

    # The tuple classes is defined in common, this is according to
    # the dataset.
    num_classes = len(classes)

    # Init empty list according to the number of classes
    correct_per_class = [0] * num_classes
    total_per_class = [0] * num_classes
    pct_per_class = [0] * num_classes
    for data, target in dataset_loader:
        with torch.no_grad():
            data = data.cuda()
            target = target.cuda()

            prediction = model(data)
            loss += F.nll_loss(prediction, target, reduction="sum")

            prediction = prediction.max(1)[1]
            correct += prediction.eq(target.view_as(prediction)).sum().item()

            #loss /= len(dataset_loader.dataset)
            total_loss = loss / len(dataset_loader.dataset)
            percentage_correct = 100.0 * correct / len(dataset_loader.dataset)
            percentage_classes = get_classes_percentage(target, prediction, classes)

            # Accumulate data from every batch in order to compute
            # the accuracy per class.
            for i in range(len(target)):
                class_id = target[i]
                if target[i] == prediction[i]:
                    correct_per_class[class_id] += 1
                total_per_class[class_id] += 1

            if (verbose):
                print("Testing set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
                    total_loss, correct, len(dataset_loader.dataset), percentage_per_class))
    percentage_per_class = compute_pct_perclass(correct_per_class, total_per_class, classes)

    return total_loss, percentage_correct, percentage_per_class


def format_model_output(e, avg_loss, tloss, testds_acc, pct_classes, classes):
    output  = "Epoch:{: <2} ".format(e)
    output += "TrainLoss:{: <4.2f} ".format(avg_loss)
    output += "TestLoss:{: <4.2f} ".format(tloss)
    output += "Accuracy:{: <5.2f}% ".format(testds_acc)

    num_classes = len(classes)
    for i in range(num_classes):
        output += "{}:{: <5.2f}% ".format(classes[i], pct_classes[i])

    return output

def build_model(
        train_loader,
        test_loader,
        epochs,
        lr,
        extra_loader,
        classes,
        model_file="PAAI21_CIFAR10_model.pt"):

    model = Model()
    model = model.cuda()

    milestones = [25, 40]

    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=milestones, gamma=0.1)

    train_model_loss = []
    testds_loss = []
    testdsT_loss = []
    print("Start train/test resnet18!")
    for epoch in range(1, epochs + 1):
        avg_loss = train_model(model, train_loader, optimizer, epoch)
        testdsL, testds_acc, testds_pcts = test_model(model, test_loader, classes)
        testdsT, testdsT_acc, testdsT_pcts = test_model(model, extra_loader, classes)

        # Collect results
        output = format_model_output(epoch, avg_loss, testdsL, testds_acc, testds_pcts, classes)
        train_model_loss.append(avg_loss)
        testds_loss.append(testdsL.item())
        testdsT_loss.append(testdsT.item())
        print(output)

        scheduler.step()

    torch.save(model.state_dict(), model_file)

    metrics = {
            "train_model_loss" : train_model_loss,
            "testds_accuracy" : testds_acc,
            "testds_loss" : testds_loss,
            "testds_classes_pcts": testds_pcts,
            "testds_accuracy_ext" : testdsT_acc,
            "testds_loss_ext" : testdsT_loss,
            "testds_classes_pcts_ext" : testdsT_pcts
    }

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


def compute_heatmap(x, model, extractor, cam_algorithm):
    img_size=x.shape
    img_size=img_size[-2:]
    if cam_algorithm == 'GradCam' or cam_algorithm == 'GradCAMpp' or cam_algorithm == 'SmoothGradCAMpp':
        scores = model(x.unsqueeze(0))
    else:
        with torch.no_grad(): scores = model(x.unsqueeze(0))

    # extract class activation map
    value, index = topk(scores, 1)
    cam = extractor(class_idx=index.item(), scores=scores)

    # Add 2 singleton dimensions to do the interpolation and adjust
    # the activation map to the image size. Then, after interpolation
    # we need to remove those extar dimensions,
    cam = cam.unsqueeze(0).unsqueeze(0)
    heat_map = F.interpolate(cam, size=img_size, mode='bilinear')
    heat_map = heat_map.squeeze(0).squeeze(0)
    cam = cam.squeeze(0).squeeze(0)

    return cam, heat_map, index, value


def crop_preprocess(x, model, extractor, cam_algorithm, cropped_pixels):
    # The 5% of the highest values represent a one of the most
    # important values for classification. These values will be
    # for experiments modified.
    replacement = 1.0
    x = x.to('cuda')
    cam_img, heat_map, index, value = compute_heatmap(x, model, extractor, cam_algorithm)
    percentile = 95
    h, w = heat_map.shape
    feature_thld = torch.quantile(heat_map, percentile * 0.01)
    heat_mask = heat_map.clone()
    bk_mask = heat_map.clone()
    heat_mask = torch.where(heat_mask >= feature_thld, 0.0, 1.0)
    bk_mask =  torch.where(bk_mask >= feature_thld, replacement, 0.0) # Replace value
    pxmes = torch.where(x == replacement)
    npixels = len(pxmes[0]) / 3 # chanles 0, 1, 2, ... has the same value
    x = torch.mul(x, heat_mask)
    x = x + bk_mask
    pxmes = torch.where(x == replacement)
    npixels = (len(pxmes[0]) / 3) - npixels
    cropped_pixels.append(npixels)

    return x.float()


def save_sample(original, cam, heat_map, crop, tgt, predt, out, prefix, i, exl):
    vcam = cam.cpu().data.numpy()
    vheat_map = heat_map.cpu().data.numpy()
    res = "FAIL!"
    if tgt == predt:
        res = "CORRECT!"

    plt.clf()
    f, ax = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
    f.suptitle('{}.{} - EXTRACTOR: {}  -  TARGET: {}  PREDICTION: {} - {}'.format(
                prefix,
                i,
                exl,
                tgt,
                predt,
                res),
                fontsize=20
    )

    # Plot
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


def save_random_samples(model_base, extractor, cam_algorithm, num_samples, crop_transformation, test_dataset, prefix, test_transform, classes):
    for i in range(num_samples):
        # CAM
        image, label =  get_one_random_sample(test_dataset)
        image_tensor = test_transform(image)
        image_tensor = image_tensor.to('cuda')
        cam_img, heat_map, index, value = compute_heatmap(image_tensor, model_base, extractor, cam_algorithm)
        cropped_image = crop_transformation(image)
        prediction = classes[index.item()]
        target = classes[label]
        out = "../res/random_sample{}.{}.png".format(prefix, i)
        save_sample(image, cam_img, heat_map, cropped_image,
                target, prediction, out, prefix, i, cam_algorithm)


def save_sequential_samples(model_base, extractor, cam_algorithm, num_samples, crop_transformation, loader, prefix, classes):
    num_samples= 10
    data_iter = iter(loader)
    images, labels = data_iter.next()

    for i in range(num_samples):
        image = images[i]
        image = image.to('cuda')
        cam_img, heat_map, index, value = compute_heatmap(image, model_base, extractor, cam_algorithm)
        image = image.cpu().detach().numpy()
        image = np.transpose(image, (1,2,0))
        cropped_image = crop_transformation(image)
        prediction = classes[index.item()]
        target = classes[labels[i]]
        out = "../res/seq_sample{}.{}.png".format(prefix, i)
        save_sample(image, cam_img, heat_map, cropped_image,
                target, prediction, out, prefix, i, cam_algorithm)


def create_new_dataset(dset, new_data, crop_transformation, train=True):
    row_list = []
    count = 0
    out_type = 'train'
    if train == False:
        out_type = 'test'
    i = 0
    outfile = new_data + out_type + '.msgpack'
    count = 0
    writer = FileWriter(outfile, len(dset), overwrite=True)

    lbls = set()
    for n, (data, label) in tqdm(enumerate(dset), ascii=True, ncols=100):
        data = crop_transformation(data)
        data = transforms.ToPILImage()(data).convert("RGB")
        bio = io.BytesIO()
        data.save(bio, 'PNG')
        image = bio.getvalue()
        writer.write(ImageClassificationData(f'{n:05d}', image, int(label)))
        lbls.update([label])
        if n == 100:
            break
    writer.close()


def save_dataset_samples(loader, out):
    num_imgs_toshow= 10
    data_iter = iter(loader)
    images, labels = data_iter.next()
    # convert images to numpy for display
    images = images.cpu().numpy()

    # plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(15, 5))
    for i in np.arange(num_imgs_toshow):
        ax = fig.add_subplot(2, num_imgs_toshow/2, i + 1, xticks=[], yticks=[])
        plt.imshow(np.transpose(images[i], (1, 2, 0)))
    plt.savefig(out, bbox_inches='tight')

def get_extractor(cam_algorithm, model):
    if cam_algorithm == 'CAM':
        extractor = CAM(model, 'resnet.layer4', 'resnet.fc')

    elif cam_algorithm == 'GradCAM':
        extractor = GradCAM(model, 'resnet.layer4', 'resnet.fc')

    elif cam_algorithm == 'GradCAMpp':
        extractor = GradCAMpp(model, 'resnet.layer4', 'resnet.fc')

    elif cam_algorithm == 'SmoothGradCAMpp':
        extractor = SmoothGradCAMpp(model, 'resnet.layer4', 'resnet.fc')

    elif cam_algorithm == 'ScoreCAM':
        extractor = ScoreCAM(model, 'resnet.layer4', 'resnet.fc')

    elif cam_algorithm == 'SSCAM':
        extractor = SSCAM(model, 'resnet.layer4', 'resnet.fc')

    elif cam_algorithm == 'ISCAM':
        extractor = ISCAM(model, 'resnet.layer4', 'resnet.fc')

    else:
        None

    return extractor
