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


def compute_centroids(img_loader):
    white_px = (255, 255, 255)
    for img, label in img_loader:
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
        print("Label:", label.item(), x_center, y_center)


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
    compute_centroids(img_loader)

if __name__ == '__main__':
    main()
