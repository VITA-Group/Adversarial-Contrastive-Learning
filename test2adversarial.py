from __future__ import print_function
import os
import argparse
import torch
from torch.nn import functional as F
from torchvision import datasets, transforms
from data.cifar10_c import CIFAR10C

from models.resnet import resnet18
from utils import logger

parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
parser.add_argument('experiment', type=str, help='exp name')
parser.add_argument('--checkpoint', default='', type=str,
                    help='path to pretrained model')
parser.add_argument('--attack_type', default='', help='which atk type to eval against, one of \
                                                     brightness, defocus_blur, fog, gaussian_blur, glass_blur, jpeg_compression, \
                                                     motion_blur, saturate, snow, speckle_noise, contrast, elastic_transform, frost,\
                                                     gaussian_noise, impulse_noise, pixelate, shot_noise, spatter, zoom_blur, transform, flowSong')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--data', type=str, default='../../data/CIFAR-10-C',
                    help='path to dir of data')


args = parser.parse_args()

# settings
model_dir = os.path.join('checkpoints_atk', args.experiment)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
log = logger(os.path.join(model_dir))
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


def eval_test(model, device, loader, log):
    model.eval()
    test_loss = 0
    correct = 0
    whole = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)

            output = model.eval()(data)
            loss = F.cross_entropy(output, target, reduction='none')
            # print("loss attacked is {}".format(loss))
            test_loss += loss.sum().item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            whole += len(target)
    test_loss /= len(loader.dataset)
    log.info('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, whole,
        100. * correct / whole))
    test_accuracy = correct / whole
    return test_loss, test_accuracy * 100


def main():
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    test_datasets = CIFAR10C(root=args.data, transform=transform_test, attack_type=args.attack_type)
    test_loader = torch.utils.data.DataLoader(
        test_datasets,
        batch_size=args.batch_size,
        shuffle=False)

    # init model, ResNet18() can be also used here for training
    model = resnet18(num_classes=10).to(device)

    assert args.checkpoint != ''

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    log.info('read checkpoint {}'.format(args.checkpoint))

    _, tacc = eval_test(model, device, test_loader, log)
    log.info("For attack type {}, tacc is {}".format(args.attack_type, tacc))
    return


if __name__ == '__main__':
    main()
