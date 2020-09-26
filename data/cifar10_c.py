import torch
from torch.utils.data import Dataset
from PIL import Image
from os.path import join
import numpy as np
from torchvision import transforms


class CIFAR10C(Dataset):
    def __init__(self, root, transform=None, severity=5, attack_type=''):
        dataPath = join(root, '{}.npy'.format(attack_type))
        labelPath = join(root, 'labels.npy')

        self.data = np.load(dataPath) #[(severity - 1) * 10000: severity * 10000]
        self.label = np.load(labelPath).astype(np.long) #[(severity - 1) * 10000: severity * 10000]
        self.transform = transform

        # print('data shape is {}'.format(self.data.shape))
        # print('label shape is {}'.format(self.label.shape))

    def __getitem__(self, idx):

        img = self.data[idx]
        img = Image.fromarray(img).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        label = self.label[idx]

        return img, label

    def __len__(self):
        return self.data.shape[0]