from __future__ import print_function
import os
import argparse
import torch
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
from utils import NormalizeByChannelMeanStd, setup_seed

from models.resnet import resnet18
from trades import trades_loss
import time
from utils import AverageMeter, eval_adv_test, logger
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import copy

parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
parser.add_argument('experiment', type=str, help='exp name')
parser.add_argument('--data', type=str, default='../../data', help='location of the data')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset to be used (cifar10 or cifar100)')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=25, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=2e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', type=float, default=8. / 255.,
                    help='perturbation')
parser.add_argument('--num-steps-train', type=int, default=10,
                    help='perturb number of steps')
parser.add_argument('--num-steps-test', type=int, default=20,
                    help='perturb number of steps')
parser.add_argument('--step-size', type=float, default=2. / 255.,
                    help='perturb step size')
parser.add_argument('--beta', type=float, default=6.0,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-freq', '-s', default=1, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--eval-only', action='store_true',
                    help='if specified, eval the loaded model')
parser.add_argument('--trainmode', default='adv', type=str,
                    help='adv or normal or test')
parser.add_argument('--fixmode', default='f1', type=str,
                    help='f1: fix nothing, f2: fix previous 3 stages, f3: fix all except fc')
parser.add_argument('--fixbn', action='store_true',
                    help='if specified, fix bn for the layers been fixed')
parser.add_argument('--use_advbn', action='store_true',
                    help='if specified, assign bn value with adv_bn value')
parser.add_argument('--checkpoint', default='', type=str,
                    help='path to pretrained model')
parser.add_argument('--resume', action='store_true',
                    help='if resume training')
parser.add_argument('--test_adv', action='store_true',
                    help='if test adv in normal mode')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='the start epoch number')
parser.add_argument('--decreasing_lr', default='15,20', help='decreasing strategy')
parser.add_argument('--cvt_state_dict', action='store_true', help='use for ss model')
parser.add_argument('--bnNameCnt', default=-1, type=int)
parser.add_argument('--trainset', type=str, default='train_idx_std',
                    help='set to be employed for training')

args = parser.parse_args()

# settings
model_dir = os.path.join('checkpoints_trade', args.experiment)
if not os.path.exists(model_dir):
  os.makedirs(model_dir)
log = logger(os.path.join(model_dir))
use_cuda = not args.no_cuda and torch.cuda.is_available()
setup_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# setup data loader
transform_train = transforms.Compose([
  transforms.RandomCrop(32, padding=4),
  transforms.RandomHorizontalFlip(),
  transforms.ToTensor(),
])
transform_test = transforms.Compose([
  transforms.ToTensor(),
])

if args.dataset == 'cifar10':
    train_datasets = torchvision.datasets.CIFAR10(root=args.data, train=True, download=True, transform=transform_train)
    vali_datasets = torchvision.datasets.CIFAR10(root=args.data, train=True, download=True, transform=transform_test)
    testset = torchvision.datasets.CIFAR10(root=args.data, train=False, download=True, transform=transform_test)
    num_classes = 10
elif args.dataset == 'cifar100':
    train_datasets = torchvision.datasets.CIFAR100(root=args.data, train=True, download=True, transform=transform_train)
    vali_datasets = torchvision.datasets.CIFAR100(root=args.data, train=True, download=True, transform=transform_test)
    testset = torchvision.datasets.CIFAR100(root=args.data, train=False, download=True, transform=transform_test)
    num_classes = 100
else:
    print("dataset {} is not supported".format(args.dataset))
    assert False


train_idx = list(np.load('split/{}.npy'.format(args.trainset)))
valid_idx = list(np.load('split/valid_idx_std.npy'))

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = torch.utils.data.DataLoader(
  train_datasets,
  batch_size=args.batch_size, sampler=train_sampler)

vali_loader = torch.utils.data.DataLoader(
  vali_datasets,
  batch_size=args.batch_size, sampler=valid_sampler)

test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)


def train(args, model, device, train_loader, optimizer, epoch, log):
  model.train()

  dataTimeAve = AverageMeter()
  totalTimeAve = AverageMeter()
  end = time.time()

  for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    dataTime = time.time() - end
    dataTimeAve.update(dataTime)

    optimizer.zero_grad()

    # calculate robust loss
    loss = trades_loss(model=model,
                       x_natural=data,
                       y=target,
                       optimizer=optimizer,
                       step_size=args.step_size,
                       epsilon=args.epsilon,
                       perturb_steps=args.num_steps_train,
                       beta=args.beta,
                       trainmode=args.trainmode,
                       fixbn=args.fixbn,
                       fixmode=args.fixmode)
    loss.backward()
    optimizer.step()

    totalTime = time.time() - end
    totalTimeAve.update(totalTime)
    end = time.time()
    # print progress
    if batch_idx % args.log_interval == 0:
      log.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tData time: {:.3f}\tTotal time: {:.3f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
               100. * batch_idx / len(train_loader), loss.item(), dataTimeAve.avg, totalTimeAve.avg))


def eval_train(model, device, train_loader, log):
  model.eval()
  train_loss = 0
  correct = 0
  whole = 0
  with torch.no_grad():
    for data, target in train_loader:
      data, target = data.to(device), target.to(device)
      output = model.eval()(data)
      train_loss += F.cross_entropy(output, target, size_average=False).item()
      pred = output.max(1, keepdim=True)[1]
      correct += pred.eq(target.view_as(pred)).sum().item()
      whole += len(target)
  train_loss /= len(train_loader.dataset)
  log.info('Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
    train_loss, correct, whole,
    100. * correct / whole))
  training_accuracy = correct / whole
  return train_loss, training_accuracy * 100


def eval_test(model, device, loader, log):
  model.eval()
  test_loss = 0
  correct = 0
  whole = 0
  with torch.no_grad():
    for data, target in loader:
      data, target = data.to(device), target.to(device)
      output = model.eval()(data)
      test_loss += F.cross_entropy(output, target, size_average=False).item()
      pred = output.max(1, keepdim=True)[1]
      correct += pred.eq(target.view_as(pred)).sum().item()
      whole += len(target)
  test_loss /= len(loader.dataset)
  log.info('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
    test_loss, correct, whole,
    100. * correct / whole))
  test_accuracy = correct / whole
  return test_loss, test_accuracy * 100


def fix_model(model, fixmode):
  if fixmode == 'f1':
    # fix none
    pass
  elif fixmode == 'f2':
    # fix previous three layers
    for name, param in model.named_parameters():
      print(name)
      if not ("layer4" in name or "fc" in name):
        print("fix {}".format(name))
        param.requires_grad = False
  elif fixmode == 'f3':
    # fix every layer except fc
    # fix previous four layers
    for name, param in model.named_parameters():
      print(name)
      if not ("fc" in name):
        print("fix {}".format(name))
        param.requires_grad = False
    pass
  else:
    assert False


def main():
  # init model, ResNet18() can be also used here for training
  model = resnet18(num_classes=num_classes).to(device)

  optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

  decreasing_lr = list(map(int, args.decreasing_lr.split(',')))
  scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)

  start_epoch = args.start_epoch

  if args.checkpoint != '':
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if 'state_dict' in checkpoint:
      state_dict = checkpoint['state_dict']
    elif 'P_state' in checkpoint:
      state_dict = checkpoint['P_state']
    else:
      state_dict = checkpoint

    if args.cvt_state_dict:
      state_dict = cvt_state_dict(state_dict, args, num_classes=num_classes)

    model.load_state_dict(state_dict)
    log.info('read checkpoint {}'.format(args.checkpoint))
    eval_test(model, device, vali_loader, log)

    if args.trainmode == 'test':
      _, tacc = eval_test(model, device, test_loader, log)
      atacc = eval_adv_test(model, device, test_loader, epsilon=args.epsilon, alpha=args.step_size, criterion=F.cross_entropy, log=log, attack_iter=args.num_steps_test)
      log.info("tacc is {}, atacc is {}".format(tacc, atacc))
      return
  elif args.resume:
    checkpoint = torch.load(os.path.join(model_dir, 'model.pt'))
    if 'state_dict' in checkpoint:
      model.load_state_dict(checkpoint['state_dict'])
    else:
      model.load_state_dict(checkpoint)

  if args.resume:
    if 'epoch' in checkpoint and 'optim' in checkpoint:
      start_epoch = checkpoint['epoch'] + 1
      optimizer.load_state_dict(checkpoint['optim'])
      for i in range(start_epoch):
        scheduler.step()
      log.info("resume the checkpoint {} from epoch {}".format(args.checkpoint, checkpoint['epoch']))
    else:
      log.info("cannot resume since lack of files")
      assert False

  if args.eval_only:
    assert args.checkpoint != ''
    _, test_tacc = eval_test(model, device, test_loader, log)
    test_atacc = eval_adv_test(model, device, test_loader, epsilon=args.epsilon, alpha=args.step_size, criterion=F.cross_entropy, log=log, attack_iter=args.num_steps_test)
    log.info("On the {}, test tacc is {}, test atacc is {}".format(args.checkpoint, test_tacc, test_atacc))
    return

  ta = []
  ata = []
  best_prec1 = 0
  best_ata = 0

  fix_model(model, args.fixmode)

  for epoch in range(start_epoch + 1, args.epochs + 1):
    # adjust learning rate for SGD
    scheduler.step()
    log.info("current lr is {}".format(optimizer.state_dict()['param_groups'][0]['lr']))

    # adversarial training
    train(args, model, device, train_loader, optimizer, epoch, log)

    # evaluation on natural examples
    print('================================================================')
    eval_train(model, device, train_loader, log)
    _, vali_tacc = eval_test(model, device, vali_loader, log)
    ta.append(vali_tacc)
    print('================================================================')

    # adv testing
    if args.trainmode != 'normal' or args.test_adv:
      vali_atacc = eval_adv_test(model, device, vali_loader, epsilon=args.epsilon, alpha=args.step_size, criterion=F.cross_entropy, log=log, attack_iter=args.num_steps_test)
    else:
      vali_atacc = 0.001
    ata.append(vali_atacc)

    # save checkpoint
    torch.save({
      'epoch': epoch,
      'state_dict': model.state_dict(),
      'optim': optimizer.state_dict(),
      'best_prec1': best_prec1,
    }, os.path.join(model_dir, 'model.pt'))

    torch.save({
      'epoch': epoch,
      'state_dict': model.state_dict(),
      'optim': optimizer.state_dict(),
      'best_prec1': best_prec1,
    }, os.path.join(model_dir, 'model_{}.pt'.format(epoch)))

    is_best = vali_tacc > best_prec1
    best_prec1 = max(vali_tacc, best_prec1)

    ata_is_best = vali_atacc > best_ata
    best_ata = max(vali_atacc, best_ata)

    if is_best:
      torch.save({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optim': optimizer.state_dict(),
        'best_prec1': best_prec1,
      }, os.path.join(model_dir, 'best_model.pt'))

    if ata_is_best:
      torch.save({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optim': optimizer.state_dict(),
        'best_prec1': best_prec1,
      }, os.path.join(model_dir, 'ata_best_model.pt'))

  # read best_model and ata_best_model for testing
  checkpoint = torch.load(os.path.join(model_dir, 'ata_best_model.pt'))
  model.load_state_dict(checkpoint['state_dict'])
  _, test_tacc = eval_test(model, device, test_loader, log)
  test_atacc = eval_adv_test(model, device, test_loader, epsilon=args.epsilon, alpha=args.step_size, criterion=F.cross_entropy, log=log, attack_iter=args.num_steps_test)
  log.info("On the ata_best_model, test tacc is {}, test atacc is {}".format(test_tacc, test_atacc))

  checkpoint = torch.load(os.path.join(model_dir, 'best_model.pt'))
  model.load_state_dict(checkpoint['state_dict'])
  _, test_tacc = eval_test(model, device, test_loader, log)
  test_atacc = eval_adv_test(model, device, test_loader, epsilon=args.epsilon, alpha=args.step_size, criterion=F.cross_entropy, log=log, attack_iter=args.num_steps_test)
  log.info("On the best_model, test tacc is {}, test atacc is {}".format(test_tacc, test_atacc))


def cvt_state_dict(state_dict, args, num_classes):
  # deal with adv bn
  state_dict_new = copy.deepcopy(state_dict)

  if args.bnNameCnt >= 0:
    for name, item in state_dict.items():
      if 'bn' in name:
        assert 'bn_list' in name
        state_dict_new[name.replace('.bn_list.{}'.format(args.bnNameCnt), '')] = item
  elif args.use_advbn:
    bn_adv_show = False
    for name, item in state_dict.items():
      if 'bn' in name and 'adv' in name:
        bn_adv_show = True
        state_dict_new[name.replace('_adv', '')] = item
    if not bn_adv_show:
      print("There no bn adv")
      assert False

  name_to_del = []
  for name, item in state_dict_new.items():
    # print(name)
    if 'bn' in name and 'adv' in name:
      name_to_del.append(name)
    if 'bn_list' in name:
      name_to_del.append(name)
    if 'fc' in name:
      name_to_del.append(name)
  for name in np.unique(name_to_del):
    del state_dict_new[name]

  # deal with down sample layer
  keys = list(state_dict_new.keys())[:]
  name_to_del = []
  for name in keys:
      if 'downsample.conv' in name:
          state_dict_new[name.replace('downsample.conv', 'downsample.0')] = state_dict_new[name]
          name_to_del.append(name)
      if 'downsample.bn' in name:
          state_dict_new[name.replace('downsample.bn', 'downsample.1')] = state_dict_new[name]
          name_to_del.append(name)
  for name in np.unique(name_to_del):
      del state_dict_new[name]

  # zero init fc
  state_dict_new['fc.weight'] = torch.zeros(num_classes, 512).to(state_dict['conv1.weight'].device)
  state_dict_new['fc.bias'] = torch.zeros(num_classes).to(state_dict['conv1.weight'].device)

  return state_dict_new


if __name__ == '__main__':
  main()
