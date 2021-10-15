import os.path
import io
import matplotlib.pyplot as plt
import PIL
import torch
import sys
from tqdm import tqdm

from datadings import reader as ddreader
from datadings.sets import ImageClassificationData
from datadings.writer import FileWriter
from datadings import reader as ddreader
from matplotlib.pyplot import imshow
from multiprocessing import Lock
from PIL import Image
from torchvision import models, datasets, transforms
from torch.utils.data import random_split, Dataset, Subset

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


def save_sample(idx, img, mask):
    plt.clf()
    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    f.suptitle('mask calculation', fontsize=20)
    out = "../res/mask_sample_{}.png".format(idx)
    # Plot
    ax[0].imshow(img)
    ax[0].set_title("Original")
    ax[1].imshow(mask)
    ax[1].set_title("Mask")
    plt.savefig(out, bbox_inches='tight')


def mask_calculation(x):
    x = x.to('cuda')
    original = x.clone()
    original = torch.where(original == 1.0, 1.0, 0.0)
    original[0] = original[0] * original[1] * original[2]
    original[1] = original[0] * original[1]
    original[2] = original[1] * original[2]

    return original.float()


def extract_masks(dset, input_dataset, out_path, train=True):
    mask_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: mask_calculation(x))
    ])

    input_dataset = input_dataset.replace('/', '')
    input_dataset = input_dataset.replace('.', '')
    out_filename = "../res/centroids_{}.json".format(input_dataset)

    count = 0
    out_type = 'train'
    if train == False:
        out_type = 'test'
    i = 0
    outfile = out_path + out_type + '.msgpack'
    count = 0
    writer = FileWriter(outfile, len(dset), overwrite=True)

    lbls = set()
    for n, (data, label) in tqdm(enumerate(dset), ascii=True, ncols=100):
        count+=1
        image = data.copy()
        data = mask_transformation(data)
        data = transforms.ToPILImage()(data).convert("RGB")

        if count <= 5:
            print("Saving....")
            save_sample(count, image, data)

        # Save new dataset
        bio = io.BytesIO()
        data.save(bio, 'PNG')
        image = bio.getvalue()
        writer.write(ImageClassificationData(f'{n:05d}', image, int(label)))
        lbls.update([label])
        #if n == 1000:
        #    break
    writer.close()
    print("TOTAL PROCESSED: ", count)





def main():
    if len(sys.argv) <= 1:
        print("error: no input directory")
        sys.exit(0)

    input_dataset = sys.argv[1]
    if os.path.isdir(input_dataset) == False:
        sys.exit("ERROR: imaga dataset directory no found")

    out_newds_dir = "mask_{}/".format(input_dataset)
    train_dataset = croppedDataset(root=input_dataset, transform=None)
    test_dataset = croppedDataset(root=input_dataset, transform=None, train=False)
    train_loader = torch.utils.data.DataLoader(train_dataset)

    print("Extract train masks and save them in: ", out_newds_dir)
    extract_masks(train_dataset, input_dataset, out_newds_dir)

    print("Extract test masks and save them in: ", out_newds_dir)
    extract_masks(test_dataset, input_dataset, out_newds_dir, train=False)


if __name__ == '__main__':
    main()
