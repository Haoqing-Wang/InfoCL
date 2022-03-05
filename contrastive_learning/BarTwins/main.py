import os
import torch
import torchvision
import torch.nn.functional as F
import argparse
import numpy as np

from model import load_model, save_model
from modules import InfoNCE, Transforms, Transforms_imagenet

parser = argparse.ArgumentParser(description='BarTwins')
parser.add_argument('--model', default="orig", type=str, help="orig/RC/LBE")
parser.add_argument('--batch_size', default=256, type=int, metavar='B', help='training batch size')
parser.add_argument('--workers', default=12, type=int, help='workers')
parser.add_argument('--epochs', default=200, type=int, help='epochs')
parser.add_argument('--save_freq', default=20, type=int, help='save frequency')
parser.add_argument('--resnet', default="resnet18", type=str, help="resnet")
parser.add_argument('--normalize', default=True, action='store_true', help='normalize')
parser.add_argument('--projection_dim', default=1024, type=int, help='projection_dim')
parser.add_argument('--lamb', default=1., type=float, help='weight of regularization term')
parser.add_argument('--zeta', default=0.1, type=float, help='variance')
parser.add_argument('--beta', default=1e-2, type=float, help='beta')
parser.add_argument('--optimizer', default="Adam", type=str, help="optimizer")
parser.add_argument('--lr', default=3e-4, type=float, help='lr')
parser.add_argument('--weight_decay', default=1e-6, type=float, help='weight_decay')
parser.add_argument('--gpus', default=4, type=int, help='number of gpu')
parser.add_argument('--model_dir', default='output/checkpoint/', type=str, help='model save path')
parser.add_argument('--dataset', default='CIFAR10', help='[CIFAR10, CIFAR100, STL-10, ImageNet]')
args = parser.parse_args()

def criterion(Z1, Z2):  # (N, d)
    N, d = Z1.size()
    Z1_norm = (Z1-Z1.mean(0, keepdim=True))/Z1.std(0, keepdim=True)
    Z2_norm = (Z2-Z2.mean(0, keepdim=True))/Z2.std(0, keepdim=True)
    C = torch.mm(Z1_norm.transpose(0, 1), Z2_norm)/N

    I = torch.eye(d).cuda()
    mask = args.beta*(1.-I)+I
    C_diff = (C-I).pow(2)
    loss = (mask*C_diff).sum()
    return loss

def train(train_loader, model, recon, optimizer):
    loss_epoch = 0.
    for step, ((x_i, x_j), _) in enumerate(train_loader):
        optimizer.zero_grad()
        if args.model == 'orig':
            _, z_i = model(x_i.cuda())
            _, z_j = model(x_j.cuda())
            loss = criterion(z_i, z_j)
        elif args.model == 'RC':
            h_i, z_i = model(x_i.cuda())
            h_j, z_j = model(x_j.cuda())
            recon_loss = F.mse_loss(recon(h_i), x_i.cuda()) + F.mse_loss(recon(h_j), x_j.cuda())
            loss = criterion(z_i, z_j) + args.lamb * recon_loss
        elif args.model == 'LBE':
            mu2_i, mu3_i, mu4_i, h2_i, h3_i, h4_i, z_i = model(x_i.cuda())
            mu2_j, mu3_j, mu4_j, h2_j, h3_j, h4_j, z_j = model(x_j.cuda())
            mu2, h2 = torch.cat([mu2_i, mu2_j], dim=0), torch.cat([h2_i, h2_j], dim=0)
            mu3, h3 = torch.cat([mu3_i, mu3_j], dim=0), torch.cat([h3_i, h3_j], dim=0)
            mu4, h4 = torch.cat([mu4_i, mu4_j], dim=0), torch.cat([h4_i, h4_j], dim=0)
            if args.dataset == "ImageNet":
                MI_estimitor = InfoNCE(mu4, h4)
            else:
                MI_estimitor = 0.25 * InfoNCE(mu2, h2) + 0.50 * InfoNCE(mu3, h3) + InfoNCE(mu4, h4)
            loss = criterion(z_i, z_j) - args.lamb * MI_estimitor
        else:
            assert False
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"Step [{step}/{len(train_loader)}]\t Loss: {loss.item()}")
        loss_epoch += loss.item()
    return loss_epoch

def main():
    data = 'non_imagenet'
    root = "../datasets"
    if args.dataset == "CIFAR10":
        train_dataset = torchvision.datasets.CIFAR10(root, download=True, transform=Transforms())
    elif args.dataset == "CIFAR100":
        train_dataset = torchvision.datasets.CIFAR100(root, download=True, transform=Transforms())
    elif args.dataset == "STL-10":
        train_dataset = torchvision.datasets.STL10(root, split='unlabeled', download=True, transform=Transforms(64))
    elif args.dataset == "ImageNet":
        traindir = os.path.join(root, 'ImageNet/train')
        train_dataset = torchvision.datasets.ImageFolder(traindir, Transforms_imagenet(size=224))
        data = 'imagenet'
    else:
        raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        sampler=None)

    log_dir = "output/log/" + args.dataset + '_%s/'%args.model
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    suffix = args.dataset + '_{}_batch_{}'.format(args.resnet, args.batch_size)
    suffix = suffix + '_proj_dim_{}'.format(args.projection_dim)
    test_log_file = open(log_dir + suffix + '.txt', "w") 

    model, recon, optimizer, scheduler = load_model(args, data=data)
    if args.dataset=='ImageNet':
        model = torch.nn.DataParallel(model, device_ids=list(range(args.gpus)))
    args.model_dir = args.model_dir + args.dataset + '_%s/'%args.model
    if not os.path.isdir(args.model_dir):
        os.mkdir(args.model_dir)
            
    for epoch in range(args.epochs):
        loss_epoch = train(train_loader, model, recon, optimizer)
        if scheduler:
            scheduler.step()
        if (epoch+1) % args.save_freq == 0:
            save_model(args.model_dir+suffix, model, epoch+1)

        print('Epoch {} loss: {}\n'.format(epoch, loss_epoch/len(train_loader)))
        print('Epoch {} loss: {}'.format(epoch, loss_epoch/len(train_loader)), file=test_log_file)
        test_log_file.flush()
    save_model(args.model_dir+suffix, model, args.epochs)

if __name__ == "__main__":
    main()