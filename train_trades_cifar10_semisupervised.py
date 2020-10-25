from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data.dataset import Dataset
from torch.optim.lr_scheduler import MultiStepLR

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

from trades import reset_model
from pdb import set_trace

from models.resnet import resnet18
from utils import pgd_attack
from utils import logger, setup_seed, AverageMeter, eval_adv_test

import time
from PIL import Image
import copy
from torch.autograd import Variable

from pdb import set_trace


class psudoSoftLabel_CIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, model, **kwds):
        super().__init__(**kwds)

        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.label = []
        # generate psudo label
        imgs = []
        for cnt, img in enumerate(self.data):
            img = Image.fromarray(img).convert('RGB')
            img = transform_test(img)
            img = img.cuda()
            imgs.append(img)

            if cnt % 100 == 99:
                imgs = torch.stack(imgs)
                print("generating psudo label {}/{}".format(cnt, len(self.data)))
                pred = model.eval()(imgs)
                self.label += pred.cpu().detach().numpy().tolist()
                imgs = []

        print("len self.label is {}".format(len(self.label)))

    def __getitem__(self, idx):
        img = self.data[idx]
        img = Image.fromarray(img).convert('RGB')
        img = self.transform(img)

        psudoLabel = torch.FloatTensor(self.label[idx])
        real_label = self.targets[idx]

        return img, psudoLabel, real_label


class distillCrossEntropy(nn.Module):
    def __init__(self, T):
        super(distillCrossEntropy, self).__init__()
        self.T = T
        return

    def forward(self, inputs, target):
        """
        :param inputs: prediction logits
        :param target: target logits
        :return: loss
        """
        log_likelihood = - F.log_softmax(inputs / self.T, dim=1)
        sample_num, class_num = target.shape
        loss = torch.sum(torch.mul(log_likelihood, torch.softmax(target / self.T, dim=1)))/sample_num

        return loss


def trade_loss_soft(model,
                    x_natural,
                    y,
                    y_soft,
                    optimizer,
                    step_size=0.003,
                    epsilon=0.031,
                    perturb_steps=10,
                    beta=1.0,
                    distance='l_inf',
                    trainmode='adv',
                    flag_adv=None,
                    T=0,
                    alpha=0,
                    rate_distill=1):
    if trainmode == "adv":
        # define KL-loss
        criterion_kl = nn.KLDivLoss(size_average=False)
        criterion_distill = distillCrossEntropy(T=T)
        model.eval()
        batch_size = len(x_natural)
        # generate adversarial example
        y_atk = torch.cat([y, y_soft[y.shape[0]:].max(1)[1]])
        x_adv = pgd_attack(model, x_natural, y_atk, None, alpha=step_size, eps=epsilon, iters=perturb_steps)

    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    model.train()
    assert trainmode == "adv"

    if flag_adv is None:
        logits = model(x_natural)
    else:
        logits = model(x_natural, flag_adv)

    if flag_adv is None:
        logits_adv = model(x_adv)
    else:
        logits_adv = model(x_adv, flag_adv)

    loss = F.cross_entropy(logits_adv[:y.shape[0]], y, reduction='sum') * (1. - alpha) / batch_size + \
           criterion_distill(logits_adv[:y.shape[0]], y_soft[:y.shape[0]]) * y_soft[:y.shape[0]].shape[0] * (T * T * rate_distill * alpha) / batch_size + \
           criterion_distill(logits_adv[y.shape[0]:], y_soft[y.shape[0]:]) * y_soft[y.shape[0]:].shape[0] * (T * T * rate_distill) / batch_size


    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(logits_adv, dim=1),
                                                    F.softmax(logits, dim=1))
    loss += beta * loss_robust

    return loss


def train_epoch(epoch, model, labeled_loader, psudoSoftLabel_CIFAR10, device, optimizer, scheduler, log):
    model.train()
    scheduler.step()
    log.info("current lr is {}".format(optimizer.state_dict()['param_groups'][0]['lr']))

    dataTimeAve = AverageMeter()
    totalTimeAve = AverageMeter()
    end = time.time()

    psudolabeled_enum = enumerate(psudoSoftLabel_CIFAR10)
    # print("len labeled_loader is {}".format(len(next(iter(labeled_loader)))))
    for batch_idx, (data, softTarget, target) in enumerate(labeled_loader):
        batch_idx_unlabeled, (inputs_unlabeled, psudoSoftTarget, _) = next(psudolabeled_enum)
        data, softTarget = torch.cat([data, inputs_unlabeled], dim=0), torch.cat([softTarget, psudoSoftTarget], dim=0)
        data, softTarget, target = data.to(device), softTarget.to(device), target.to(device)

        # print("acc is {}".format((softTarget[:target.shape[0]].max(1)[1] == target).float().mean()))

        dataTime = time.time() - end
        dataTimeAve.update(dataTime)

        optimizer.zero_grad()

        # calculate robust loss
        loss = trade_loss_soft(model=model,
                               x_natural=data,
                               y=target,
                               y_soft=softTarget,
                               optimizer=optimizer,
                               step_size=args.step_size,
                               epsilon=args.epsilon,
                               perturb_steps=args.num_steps_train,
                               beta=args.beta,
                               trainmode='adv',
                               T=args.T,
                               alpha=args.alpha,
                               rate_distill=args.rate_distill)

        loss.backward()
        optimizer.step()

        totalTime = time.time() - end
        totalTimeAve.update(totalTime)
        end = time.time()
        # print progress
        if batch_idx % args.log_interval == 0:
            log.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tData time: {:.3f}\tTotal time: {:.3f}'.format(
                epoch, batch_idx, len(labeled_loader),
                100. * batch_idx / len(labeled_loader), loss.item(), dataTimeAve.avg, totalTimeAve.avg))


def cvt_state_dict(state_dict, args, num_classes=10):
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='CIFAR-10: semi-supervised experiments')
    parser.add_argument('experiment', type=str, help='path to saving the model')
    parser.add_argument('--data', type=str, default='../../data', help='location of the data')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--decreasing_lr', default='15,20', help='decreasing strategy')
    parser.add_argument('--checkpoint', default='', help='checkpoint')
    parser.add_argument('--checkpoint_clean', default='', help='the model used for generate psudo labels')
    parser.add_argument('--use_advbn', action='store_true',
                        help='if specified, assign bn value with adv_bn value')
    parser.add_argument('--cvt_state_dict', action='store_true', help='use for ss model')
    parser.add_argument('--resume', action='store_true', help='checkpoint')
    parser.add_argument('--nlabel', default=5000, type=int, help='number of labels')
    parser.add_argument('--epochs', default=25, type=int, help='number of labels')
    parser.add_argument('--seed', default=10, type=int, help='seed')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--beta', type=float, default=6.0,
                        help='regularization, i.e., 1/lambda in TRADES')
    parser.add_argument('--epsilon', type=float, default=8. / 255.,
                        help='perturbation')
    parser.add_argument('--num-steps-train', type=int, default=10,
                        help='perturb number of steps')
    parser.add_argument('--num-steps-test', type=int, default=20,
                        help='perturb number of steps')
    parser.add_argument('--step-size', type=float, default=2. / 255.,
                        help='perturb step size')
    parser.add_argument('--alpha', type=float, default=0.5, help='perturb step size')
    parser.add_argument('--rate_distill', type=float, default=1.0, help='the rate for distillation loss')
    parser.add_argument('--T', type=float, default=1.0,
                        help='perturb step size')
    parser.add_argument('--percentageLabeledData', type=int, default=10, help='the percentage of labeled data, choose between [1, 10]')

    parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--bnNameCnt', default=-1, type=int)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    save_dir = os.path.join('checkpoints_semi', args.experiment)
    if os.path.exists(save_dir) is not True:
        os.system("mkdir -p {}".format(save_dir))

    log = logger(path=save_dir)
    log.info(str(args))
    setup_seed(args.seed)

    # read psudo label generating model
    gene_net = resnet18(num_classes=10)
    gene_net.load_state_dict(torch.load(args.checkpoint_clean)['state_dict'], strict=False)
    gene_net = gene_net.to(device)

    # Prepare data
    log.info('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_datasets = torchvision.datasets.CIFAR10(root=args.data, train=True, download=True, transform=transform_train)
    train_datasets_psudolabeled = psudoSoftLabel_CIFAR10(root=args.data, train=True, download=True, transform=transform_train, model=gene_net)
    vali_datasets = torchvision.datasets.CIFAR10(root=args.data, train=True, download=True, transform=transform_test)
    test_datasets = torchvision.datasets.CIFAR10(root=args.data, train=False, download=True, transform=transform_test)

    valid_idx = list(np.load('split/valid_idx_std.npy'))
    if args.percentageLabeledData == 10:
        label_idx = list(np.load('split/train0.1_idx.npy'))
        unlabel_idx = list(np.load('split/train0.9_unlabel_idx.npy'))
        batchWholeNum = 180
    elif args.percentageLabeledData == 1:
        label_idx = list(np.load('split/train0.01_idx.npy'))
        unlabel_idx = list(np.load('split/train0.99_idx.npy'))
        batchWholeNum = 225
    else:
        print("the precentage of {}% is not available".format(args.percentageLabeledData))
        assert False

    print("len unlabel_idx is {}".format(len(unlabel_idx)))

    label_sampler = SubsetRandomSampler(label_idx)
    unlabel_sampler = SubsetRandomSampler(unlabel_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    labeledPartloader = torch.utils.data.DataLoader(train_datasets_psudolabeled,
                                                    batch_size=(len(label_idx)//batchWholeNum),
                                                    shuffle=False, num_workers=4,
                                                    sampler=label_sampler)
    psudolabeledPartloader = torch.utils.data.DataLoader(train_datasets_psudolabeled,
                                                         batch_size=(len(unlabel_idx)//batchWholeNum),
                                                         shuffle=False, num_workers=4,
                                                         sampler=unlabel_sampler)


    print("len labeled_loader is {}, len psudolabeled_loader is {}".format(len(labeledPartloader), len(psudolabeledPartloader)))

    # test psudo label acc
    test_datasets_psudolabeled = psudoSoftLabel_CIFAR10(root=args.data, train=False, download=True, transform=transform_train, model=gene_net)
    test_psudolabeled_loader = torch.utils.data.DataLoader(test_datasets_psudolabeled,
                                                           batch_size=128,
                                                           shuffle=False, num_workers=4)

    for loader, set_name in zip([test_psudolabeled_loader, psudolabeledPartloader], ['test set', 'unlabeled train set']):
      cnt, right = 0, 0
      l1_distances = AverageMeter()

      for _, psudo_label, label in loader:
          cnt += len(label)
          right += (psudo_label.max(1)[1] == label).sum().float()
          psudoLabelOnehot = F.one_hot(psudo_label.max(1)[1], num_classes=10)
          psudo_label_prob = F.softmax(psudo_label, dim=1)

      print("psudo label acc is {:.02}, cnt {}, right {} in {}".format(right/cnt, cnt, right, set_name))

    vali_loader = torch.utils.data.DataLoader(
      vali_datasets,
      batch_size=args.batch_size, sampler=valid_sampler)

    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=args.batch_size)

    # Data loader used for test
    testdata_test = torchvision.datasets.CIFAR10(root=args.data, train=False, download=True, transform=transform_test)
    testdata_test_loader = torch.utils.data.DataLoader(testdata_test,
                                                       batch_size=128,
                                                       shuffle=False, num_workers=4)

    # Build model
    log.info('==> Building model..')
    # basic_net = model.resnet32_w10()
    net = resnet18(num_classes=10)
    net = net.to(device)
    if device == 'cuda':
        # net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    # Train
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    decreasing_lr = list(map(int, args.decreasing_lr.split(',')))
    scheduler = MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)

    start_epoch = 1
    if args.checkpoint != '':
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        if 'state_dict' in checkpoint:
          state_dict = checkpoint['state_dict']
        elif 'net' in checkpoint:
          state_dict = checkpoint['net']
        else:
          state_dict = checkpoint

        if args.cvt_state_dict:
          state_dict = cvt_state_dict(state_dict, args)

        net.load_state_dict(state_dict)
        log.info('read checkpoint {}'.format(args.checkpoint))

    elif args.resume:
      checkpoint = torch.load(os.path.join(save_dir, 'model.pt'))
      if 'state_dict' in checkpoint:
          net.load_state_dict(checkpoint['state_dict'])
      else:
          net.load_state_dict(checkpoint)

    if args.resume:
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            for i in range(start_epoch):
                scheduler.step()
            log.info("resume the checkpoint {} from epoch {}".format(args.checkpoint, checkpoint['epoch']))
        else:
            log.info("cannot resume since lack of files")
            assert False

    best_prec1 = 0
    best_ata = 0
    for epoch in range(start_epoch, args.epochs + 1):
        train_epoch(epoch, net, labeledPartloader, psudolabeledPartloader,
                    device, optimizer, scheduler, log)
        _, natural_test_acc = eval_test(net, device, vali_loader, log)
        robust_test_acc = eval_adv_test(net, device, vali_loader, epsilon=args.epsilon, alpha=args.step_size, criterion=F.cross_entropy, log=log, attack_iter=args.num_steps_test)

        state = {
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optim': optimizer.state_dict(),
            'best_prec1': best_prec1,
        }
        torch.save(state, '{}/{}.pt'.format(save_dir, epoch))
        torch.save(state, '{}/model.pt'.format(save_dir))

        is_best = natural_test_acc > best_prec1
        best_prec1 = max(natural_test_acc, best_prec1)

        ata_is_best = robust_test_acc > best_ata
        best_ata = max(robust_test_acc, best_ata)

        if is_best:
            torch.save({
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'optim': optimizer.state_dict(),
                'best_prec1': best_prec1,
            }, os.path.join(save_dir, 'best_model.pt'))

        if ata_is_best:
            torch.save({
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'optim': optimizer.state_dict(),
                'best_prec1': best_prec1,
            }, os.path.join(save_dir, 'ata_best_model.pt'))

        # read best_model and ata_best_model for testing
        checkpoint = torch.load(os.path.join(save_dir, 'ata_best_model.pt'))
        net.load_state_dict(checkpoint['state_dict'])
        _, natural_test_acc = eval_test(net, device, test_loader, log)
        robust_test_acc = eval_adv_test(net, device, test_loader, epsilon=args.epsilon, alpha=args.step_size, criterion=F.cross_entropy, log=log, attack_iter=args.num_steps_test)
        log.info("On the ata_best_model, test tacc is {}, test atacc is {}".format(natural_test_acc, robust_test_acc))

        checkpoint = torch.load(os.path.join(save_dir, 'best_model.pt'))
        net.load_state_dict(checkpoint['state_dict'])
        _, natural_test_acc = eval_test(net, device, test_loader, log)
        robust_test_acc = eval_adv_test(net, device, test_loader, epsilon=args.epsilon, alpha=args.step_size, criterion=F.cross_entropy, log=log, attack_iter=args.num_steps_test)
        log.info("On the best_model, test tacc is {}, test atacc is {}".format(natural_test_acc, robust_test_acc))
