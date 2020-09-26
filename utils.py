import torch
import torch.nn as nn
import os
import time
import numpy as np
import random
import copy
from pdb import set_trace
from collections import OrderedDict


def normalize_fn(tensor, mean, std):
    """Differentiable version of torchvision.functional.normalize"""
    # here we assume the color channel is in at dim=1
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)


class NormalizeByChannelMeanStd(nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)


def pgd_attack(model, images, labels, device, eps=8. / 255., alpha=2. / 255., iters=20, advFlag=None, forceEval=True, randomInit=True):
    # images = images.to(device)
    # labels = labels.to(device)
    loss = nn.CrossEntropyLoss()

    # init
    if randomInit:
        delta = torch.rand_like(images) * eps * 2 - eps
    else:
        delta = torch.zeros_like(images)
    delta = torch.nn.Parameter(delta, requires_grad=True)

    for i in range(iters):
        if advFlag is None:
            if forceEval:
                model.eval()
            outputs = model(images + delta)
        else:
            if forceEval:
                model.eval()
            outputs = model(images + delta, advFlag)

        model.zero_grad()
        cost = loss(outputs, labels)
        # cost.backward()
        delta_grad = torch.autograd.grad(cost, [delta])[0]

        delta.data = delta.data + alpha * delta_grad.sign()
        delta.grad = None
        delta.data = torch.clamp(delta.data, min=-eps, max=eps)
        delta.data = torch.clamp(images + delta.data, min=0, max=1) - images

    model.zero_grad()

    return (images + delta).detach()


def eval_adv_test(model, device, test_loader, epsilon, alpha, criterion, log, attack_iter=40):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # fix random seed for testing
    torch.manual_seed(1)

    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(test_loader):
        input, target = input.to(device), target.to(device)
        input_adv = pgd_attack(model, input, target, device, eps=epsilon, iters=attack_iter, alpha=alpha).data

        # compute output
        output = model.eval()(input_adv)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, = accuracy(output.data, target, topk=(1,))
        top1.update(prec1, input.size(0))
        losses.update(loss.item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if (i % 10 == 0) or (i == len(test_loader) - 1):
            log.info(
                'Test: [{}/{}]\t'
                'Time: {batch_time.val:.4f}({batch_time.avg:.4f})\t'
                'Loss: {loss.val:.3f}({loss.avg:.3f})\t'
                'Prec@1: {top1.val:.3f}({top1.avg:.3f})\t'.format(
                    i, len(test_loader), batch_time=batch_time,
                    loss=losses, top1=top1
                )
            )

    log.info(' * Adv Prec@1 {top1.avg:.3f}'.format(top1=top1))
    return top1.avg


def eval_adv_test_dist(model, device, test_loader, epsilon, alpha, criterion, log, world_size, attack_iter=40, randomInit=True):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # fix random seed for testing
    torch.manual_seed(1)

    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(test_loader):
        input, target = input.cuda(non_blocking=True), target.cuda(non_blocking=True)
        input_adv = pgd_attack(model, input, target, device, eps=epsilon, iters=attack_iter, alpha=alpha, randomInit=randomInit).data

        # compute output
        output = model(input_adv)
        output_list = [torch.zeros_like(output) for _ in range(world_size)]
        torch.distributed.all_gather(output_list, output)
        output = torch.cat(output_list)

        target_list = [torch.zeros_like(target) for _ in range(world_size)]
        torch.distributed.all_gather(target_list, target)
        target = torch.cat(target_list)

        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, = accuracy(output.data, target, topk=(1,))
        top1.update(prec1, input.size(0))
        losses.update(loss.item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if (i % 10 == 0) or (i == len(test_loader) - 1):
            log.info(
                'Test: [{}/{}]\t'
                'Time: {batch_time.val:.4f}({batch_time.avg:.4f})\t'
                'Loss: {loss.val:.3f}({loss.avg:.3f})\t'
                'Prec@1: {top1.val:.3f}({top1.avg:.3f})\t'.format(
                    i, len(test_loader), batch_time=batch_time,
                    loss=losses, top1=top1
                )
            )

    log.info(' * Adv Prec@1 {top1.avg:.3f}'.format(top1=top1))
    return top1.avg

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class logger(object):
    def __init__(self, path):
        self.path = path

    def info(self, msg):
        print(msg)
        with open(os.path.join(self.path, "log.txt"), 'a') as f:
            f.write(msg + "\n")


def fix_bn(model, fixmode):
    if fixmode == 'f1':
        # fix none
        pass
    elif fixmode == 'f2':
        # fix previous three layers
        for name, m in model.named_modules():
            if not ("layer4" in name or "fc" in name):
                m.eval()
    elif fixmode == 'f3':
        # fix every layer except fc
        # fix previous four layers
        for name, m in model.named_modules():
            if not ("fc" in name):
                m.eval()
    else:
        assert False


# loss
def pair_cosine_similarity(x, eps=1e-8):
    n = x.norm(p=2, dim=1, keepdim=True)
    return (x @ x.t()) / (n * n.t()).clamp(min=eps)


def nt_xent(x, t=0.5):
    # print("device of x is {}".format(x.device))
    x = pair_cosine_similarity(x)
    x = torch.exp(x / t)
    idx = torch.arange(x.size()[0])
    # Put positive pairs on the diagonal
    idx[::2] += 1
    idx[1::2] -= 1
    x = x[idx]
    # subtract the similarity of 1 from the numerator
    x = x.diag() / (x.sum(0) - torch.exp(torch.tensor(1 / t)))
    return -torch.log(x).mean()


def cvtPrevious2bnToCurrent2bn(state_dict):
    """
    :param state_dict: old state dict with bn and bn adv
    :return:
    """
    new_state_dict = OrderedDict()
    for name, value in state_dict.items():
        if ('bn1' in name) and ('adv' not in name):
            newName = name.replace('bn1.', 'bn1.bn_list.0.')
        elif ('bn1' in name) and ('adv' in name):
            newName = name.replace('bn1_adv.', 'bn1.bn_list.1.')
        elif ('bn2' in name) and ('adv' not in name):
            newName = name.replace('bn2.', 'bn2.bn_list.0.')
        elif ('bn2' in name) and ('adv' in name):
            newName = name.replace('bn2_adv.', 'bn2.bn_list.1.')
        elif ('bn.' in name):
            newName = name.replace('bn.', 'bn.bn_list.0.')
        elif ('bn_adv.' in name):
            newName = name.replace('bn_adv.', 'bn.bn_list.1.')
        elif 'bn3' in name:
            assert False
        else:
            newName = name

        print("convert name: {} to {}".format(name, newName))
        new_state_dict[newName] = value
    return new_state_dict


class augStrengthScheduler(object):
    """Computes and stores the average and current value"""
    def __init__(self, aug_dif_scheduler_strength_range, aug_dif_scheduler_epoch_range, transGeneFun):
        if ',' in aug_dif_scheduler_strength_range:
            self.aug_dif_scheduler_strength_range = list(map(float, aug_dif_scheduler_strength_range.split(',')))
        else:
            self.aug_dif_scheduler_strength_range = []

        if ',' in aug_dif_scheduler_epoch_range:
            self.aug_dif_scheduler_epoch_range = list(map(int, aug_dif_scheduler_epoch_range.split(',')))
        else:
            self.aug_dif_scheduler_epoch_range = []
        self.transGeneFun = transGeneFun
        self.epoch = 0

        assert (len(self.aug_dif_scheduler_strength_range) == 2 and len(self.aug_dif_scheduler_epoch_range) == 2) or \
               (len(self.aug_dif_scheduler_strength_range) == 0 and len(self.aug_dif_scheduler_epoch_range) == 0)

    def step(self):
        self.epoch += 1

        if len(self.aug_dif_scheduler_strength_range) == 0 and len(self.aug_dif_scheduler_epoch_range) == 0:
            return self.transGeneFun(1.0)
        else:
            startStrength, endStrength = self.aug_dif_scheduler_strength_range
            startEpoch, endEpoch = self.aug_dif_scheduler_epoch_range
            strength = min(max(0, self.epoch - startEpoch), endEpoch - startEpoch) / (endEpoch - startEpoch) * (endStrength - startStrength) + startStrength
            return self.transGeneFun(strength)

# new_state_dict = cvtPrevious2bnToCurrent2bn(checkpoint['state_dict'])
# model.load_state_dict(new_state_dict)
