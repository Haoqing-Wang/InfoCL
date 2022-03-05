import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import argparse
import numpy as np

from model import load_model, save_model
from modules import Transforms
from modules import InfoNCE
from data_statistics import get_data_nclass

parser = argparse.ArgumentParser(description='Supervised')
parser.add_argument('--model', default="orig", type=str, help="orig/RC/LBE")
parser.add_argument('--batch_size', default=256, type=int, metavar='B', help='training batch size')
parser.add_argument('--workers', default=12, type=int, help='workers')
parser.add_argument('--epochs', default=200, type=int, help='epochs')
parser.add_argument('--save_freq', default=20, type=int, help='save frequency')
parser.add_argument('--resnet', default="resnet18", type=str, help="resnet")
parser.add_argument('--lamb', default=0.1, type=float, help='weight of regularization term')
parser.add_argument('--zeta', default=0.01, type=float, help='variance')
parser.add_argument('--lr', default=3e-4, type=float, help='learning rate')
parser.add_argument('--model_dir', default='output/', type=str, help='model save path')
parser.add_argument('--dataset', default='CIFAR10', help='[CIFAR10, CIFAR100]')
args = parser.parse_args()

def train(train_loader, model, recon, criterion, optimizer):
    loss_epoch = 0.
    for step, (x, y) in enumerate(train_loader):
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        if args.model == 'orig':
            _, pred = model(x)
            loss = criterion(pred, y)
        elif args.model == 'RC':
            h, pred = model(x)
            recon_loss = F.mse_loss(recon(h), x)
            loss = criterion(pred, y) + args.lamb * recon_loss
        elif args.model == 'LBE':
            mu2, mu3, mu4, h2, h3, h4, pred = model(x)
            MI_estimitor = 0.25 * InfoNCE(mu2, h2) + 0.50 * InfoNCE(mu3, h3) + InfoNCE(mu4, h4)
            loss = criterion(pred, y) - args.lamb * MI_estimitor
        else:
            assert False
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"Step [{step}/{len(train_loader)}]\t Loss: {loss.item()}")
        loss_epoch += loss.item()
    return loss_epoch

def main():
    root = '../datasets'
    if args.dataset == "CIFAR10":
        train_dataset = torchvision.datasets.CIFAR10(root, download=True, transform=Transforms())
    elif args.dataset == "CIFAR100":
        train_dataset = torchvision.datasets.CIFAR100(root, download=True, transform=Transforms())
    else:
        raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        sampler=None)

    log_dir = args.model_dir + "log/" + args.dataset + '_%s/'%args.model
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    suffix = args.dataset + '_{}_batch_{}'.format(args.resnet, args.batch_size)
    test_log_file = open(log_dir + suffix + '.txt', "w")

    model, recon, optimizer, scheduler = load_model(args, get_data_nclass(args.dataset))
    args.model_dir = args.model_dir + 'checkpoint/' + args.dataset + '_%s/'%args.model
    if not os.path.isdir(args.model_dir):
        os.mkdir(args.model_dir)

    criterion = nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        loss_epoch = train(train_loader, model, recon, criterion, optimizer)
        if scheduler:
            scheduler.step()
        if (epoch+1) % args.save_freq == 0:
            save_model(args.model_dir+suffix, model, epoch+1)

        print('Epoch {} loss: {}\n'.format(epoch, loss_epoch / len(train_loader)))
        print('Epoch {} loss: {}'.format(epoch, loss_epoch/len(train_loader)), file=test_log_file)
        test_log_file.flush()

    save_model(args.model_dir+suffix, model, args.epochs)

if __name__ == "__main__":
    main()