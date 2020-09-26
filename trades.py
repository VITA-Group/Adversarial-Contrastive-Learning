import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from utils import pgd_attack, fix_bn
import numpy as np


def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()


def reset_model(model, fixmode):
    if fixmode == 'f1':
        for name, param in model.named_parameters():
                param.requires_grad = True

    elif fixmode == 'f2':
        # fix previous three layers
        for name, param in model.named_parameters():
            if not ("layer4" in name or "fc" in name):
                param.requires_grad = False
            else:
                param.requires_grad = True
    elif fixmode == 'f3':
        # fix every layer except fc
        # fix previous four layers
        for name, param in model.named_parameters():
            if not ("fc" in name):
                param.requires_grad = False
            else:
                param.requires_grad = True
    else:
        assert False


def trades_loss(model,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                distance='l_inf',
                trainmode='adv',
                fixbn=False,
                fixmode='',
                flag_adv=None):
    if trainmode == "adv":
        batch_size = len(x_natural)
        # define KL-loss
        criterion_kl = nn.KLDivLoss(size_average=False)
        model.eval()

        # generate adversarial example
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
        if distance == 'l_inf':
            for _ in range(perturb_steps):
                x_adv.requires_grad_()
                with torch.enable_grad():
                    if flag_adv is None:
                        model.eval()
                        loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                               F.softmax(model(x_natural), dim=1))
                    else:
                        model.eval()
                        loss_kl = criterion_kl(F.log_softmax(model(x_adv, flag_adv), dim=1),
                                               F.softmax(model(x_natural, flag_adv), dim=1))
                grad = torch.autograd.grad(loss_kl, [x_adv])[0]
                x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
        elif distance == 'l_2':
            assert False
        else:
            assert False

        x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    model.train()

    if fixbn:
        fix_bn(model, fixmode)

    if flag_adv is None:
        logits = model(x_natural)
    else:
        logits = model(x_natural, flag_adv)
    loss = F.cross_entropy(logits, y)

    if trainmode == "adv":
        if flag_adv is None:
            logits_adv = model(x_adv)
        else:
            logits_adv = model(x_adv, flag_adv)

        loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(logits_adv, dim=1),
                                                        F.softmax(logits, dim=1))
        loss += beta * loss_robust
    return loss
