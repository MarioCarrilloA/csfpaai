import os.path
import sys
import io
import PIL
from PIL import Image
import torch
from torchvision import models, datasets, transforms
from torch.utils.data import random_split, Dataset, Subset
from multiprocessing import Lock
from datadings import reader as ddreader
import statistics as stat 
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import json

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


def save_sample(dset_name, img_id, img, label, x, y):
    plt.clf()
    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    f.suptitle('dataset: {} - Target: {}'.format(dset_name, classes[label]), fontsize=20)
    out = "../res/centroid_{}_sample_{}.png".format(dset_name, img_id)
    # Plot
    ax[0].imshow(img)
    ax[0].set_title("Original")
    ax[1].scatter(x, y)
    ax[1].set_xlim([0, 32])
    ax[1].set_ylim([32, 0])
    ax[1].set_title("Cropped image")
    plt.savefig(out, bbox_inches='tight')


def compute_centroids(img_loader, input_dataset):
    white_px = (255, 255, 255)
    img_id = 0
    max_samples = 10
    centroids_per_label = {}
    input_dataset = input_dataset.replace('/', '')
    input_dataset = input_dataset.replace('.', '')
    out_filename = "../res/centroids_{}.json".format(input_dataset)
    print("Computing centroids")
    for img, label in img_loader:
        img_id += 1
        img = img.squeeze(0)
        img = transforms.ToPILImage()(img).convert("RGB")
        count = 0
        wx = []
        wy = []
        for x in range(img.width):
            for y in range(img.height):
                px = img.getpixel((x,y))
                if px == white_px:
                    count += 1
                    wx.append(x)
                    wy.append(y)
        x_center = stat.mean(wx)
        y_center = stat.mean(wy)
        if img_id <= max_samples:
            print("Saving sample ", img_id)
            save_sample(input_dataset, img_id, img, label.item(), x_center, y_center)

        # create dictionary with data
        if label.item() in centroids_per_label:
            centroids_per_label[label.item()].append((x_center, y_center))
        else:
            tmp = [(x_center, y_center)]
            centroids_per_label.update({label.item(): tmp})

    # collect results in JSON file
    collect_results(centroids_per_label, out_filename)

def main():
    if len(sys.argv) <= 1:
        print("error: no input directory")
        sys.exit(0)

    input_dataset = sys.argv[1]
    if os.path.isdir(input_dataset) == False:
        sys.exit("ERROR: imaga dataset directory no found")

    
    img_dataset = croppedDataset(root=input_dataset, transform=transforms.ToTensor())
    #img_dataset = croppedDataset(root=input_dataset)
    img_loader = torch.utils.data.DataLoader(img_dataset)
    compute_centroids(img_loader, input_dataset)

if __name__ == '__main__':
    main()
