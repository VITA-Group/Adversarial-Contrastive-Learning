import torch
from torchvision.datasets import CIFAR10, CIFAR100
from PIL import Image

class CustomCIFAR10(CIFAR10):
    def __init__(self, withLabel=False, labelSubSet=None, labelTrans=None, **kwds):
        super().__init__(**kwds)
        self.withLabel = withLabel
        self.labelTrans = labelTrans

        if labelSubSet is not None:
            self.data = self.data[labelSubSet]

    def __getitem__(self, idx):
        # if not self.train:
        #     return super().__getitem__(idx)

        img = self.data[idx]
        img = Image.fromarray(img).convert('RGB')
        imgs = [self.transform(img), self.transform(img)]
        if not self.withLabel:
            return torch.stack(imgs)
        else:
            imgLabelTrans = self.labelTrans(img)
            label = self.targets[idx]
            return torch.stack(imgs), imgLabelTrans, label


class CustomCIFAR100(CIFAR100):
    def __init__(self, withLabel=False, labelTrans=None, **kwds):
        super().__init__(**kwds)
        self.withLabel = withLabel
        self.labelTrans = labelTrans

    def __getitem__(self, idx):
        # if not self.train:
        #     return super().__getitem__(idx)

        img = self.data[idx]
        img = Image.fromarray(img).convert('RGB')
        imgs = [self.transform(img), self.transform(img)]
        if not self.withLabel:
            return torch.stack(imgs)
        else:
            imgLabelTrans = self.labelTrans(img)
            label = self.targets[idx]
            return torch.stack(imgs), imgLabelTrans, label

