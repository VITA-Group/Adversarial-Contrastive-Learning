# Adversarial Contrastive Learning: Harvesting More Robustness from Unsupervised Pre-Training
## Introduction
Recent work has shown that, when integrated with adversarial training, self-supervised 
pre-training with several pretext tasks can lead to state-of-the-art robustness. In this 
work, we show that contrasting features to random and adversarial perturbations for consistency
 can benefit robustness-aware pre-training even further. Our approach leverages a recent 
 contrastive learning framework, which learns representations by maximizing feature consistency 
 under differently augmented views. This fits particularly well with the goal of adversarial robustness, 
 as one cause of adversarial fragility is the lack of feature invariance, i.e., small input perturbations 
 can result in undesirable large changes in features or even predicted labels. We explore various options 
 to formulate the contrastive task, and demonstrate that by injecting adversarial augmentations, 
 contrastive pre-training indeed contributes to learning data-efficient robust models. We extensively 
 evaluate the proposed Adversarial Contrastive Learning (ACL) and show it can consistently outperform state-of-the-arts. 
 For example on the CIFAR-10 dataset, ACL outperforms the latest unsupervised robust pre-training approach
  with substantial margins: 2.99% on robust accuracy and 2.14% on standard accuracy. We further demonstrate 
  that ACL pre-training can improve semi-supervised adversarial training, even at very low label rates.

## Method
![pipeline](imgs/pipeline.png)
Illustration of workflow comparison: (a) The original SimCLR framework, a.k.a., standard to standard (no adversarial attack involved); 
(b) - (d) three proposed variants of our adversarial contrastive learning framework: A2A, A2S, and DS (our best solution). 
Note that, whenever more than one encoder branches co-exist in one framework, they by default share all weights, except that adversarial and standard 
encoders will use independent BN parameters.
## Environment setup
require: pytorch==1.6.0

## Pretraining
Pretrain the model on CIFAR-10 with ACL(DS)
```bash
python train_simCLR.py ACL_DS --ACL_DS --data \path\to\data
```
Pretrain the model on CIFAR-100 with ACL(DS)
```bash
python train_simCLR.py ACL_DS_CIFAR100 --ACL_DS --dataset cifar100 --data \path\to\data
```
## Finetuning
Adversarial finetune ACL(DS) pretraining model on CIFAR-10 (Need to do ACL(DS) pretraining on CIFAR10 first)
```bash
python train_trades.py ACL_DS_TUNE --checkpoint checkpoints/ACL_DS/model_1000.pt --cvt_state_dict --bnNameCnt 1
```
Adversarial finetune ACL(DS) pretraining model on CIFAR-100 (Need to do ACL(DS) pretraining on CIFAR100 first)
```bash
python train_trades.py ACL_DS_CIFAR100_TUNE --dataset cifar100 --checkpoint checkpoints/ACL_DS_CIFAR100/model_1000.pt --cvt_state_dict --bnNameCnt 1
```
# Semi-supervised adversarial training
On CIFAR-10 with 0.01 available labels (Need to do ACL(DS) pretraining on CIFAR10 first)
```bash
# train the standard model for generating psudo labels
python train_trades.py ACL_DS_SEMI_STAGE2_0.01LABELS --trainmode normal --trainset train0.01_idx --checkpoint checkpoints/ACL_DS/model_1000.pt --cvt_state_dict --bnNameCnt 0
# Adversarial finetuning from ACL(DS) with the psudo labels
python train_trades_cifar10_semisupervised.py ACL_DS_SEMI_STAGE3_0.01LABELS --checkpoint checkpoints/ACL_DS/model_1000.pt --bnNameCnt 1 --cvt_state_dict --checkpoint_clean checkpoints_trade/ACL_DS_SEMI_STAGE2_0.01LABELS/best_model.pt --percentageLabeledData 1
```
On CIFAR-10 with 0.1 available labels (Need to do ACL(DS) pretraining on CIFAR10 first)
```bash
# train the standard model for generating psudo labels
python train_trades.py ACL_DS_SEMI_STAGE2_0.1LABELS --trainmode normal --trainset train0.1_idx --checkpoint checkpoints/ACL_DS/model_1000.pt --cvt_state_dict --bnNameCnt 0
# Adversarial finetuning from ACL(DS) with the psudo labels
python train_trades_cifar10_semisupervised.py ACL_DS_SEMI_STAGE3_0.1LABELS --checkpoint checkpoints/ACL_DS/model_1000.pt --bnNameCnt 1 --cvt_state_dict --checkpoint_clean checkpoints_trade/ACL_DS_SEMI_STAGE2_0.1LABELS/best_model.pt --percentageLabeledData 10
```

# Acknowledge
1. Trade fine-tuning code from [TRADE](https://github.com/yaodongyu/TRADES) (official code). 
2. Semi-supervised fine-tuning code partially borrow from [UAT](https://github.com/deepmind/deepmind-research/tree/master/unsupervised_adversarial_training) (official code). 
