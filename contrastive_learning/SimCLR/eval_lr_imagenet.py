import os
import argparse
import torch
import torchvision
import torchvision.transforms as transforms

from model import load_model
from modules import LogisticRegression
from data_statistics import get_data_mean_and_stdev, get_data_nclass
from transfer_datasets import DATASET
import numpy as np

parser = argparse.ArgumentParser(description='linear Evaluation for ImageNet')
parser.add_argument('--model', default="orig", type=str, help="orig/RC/LBE")
parser.add_argument('--logistic_batch_size', default=256, type=int, metavar='B', help='logistic_batch_size batch size')
parser.add_argument('--logistic_epochs', default=100, type=int, help='logistic_epochs')
parser.add_argument('--batch_size', default=1024, type=int, metavar='B', help='training batch size')
parser.add_argument('--workers', default=12, type=int, help='workers')
parser.add_argument('--epochs', default=200, type=int, help='epochs')
parser.add_argument('--resnet', default="resnet50", type=str, help="resnet")
parser.add_argument('--normalize', default=True, action='store_true', help='normalize')
parser.add_argument('--projection_dim', default=128, type=int, help='projection_dim')
parser.add_argument('--lamb', default=1., type=float, help='weight of regularization term')
parser.add_argument('--optimizer', default="LARS", type=str, help="optimizer")
parser.add_argument('--weight_decay', default=1e-6, type=float, help='weight_decay')
parser.add_argument('--lr', default=0.3, type=float, help='lr')
parser.add_argument('--temperature', default=0.1, type=float, help='temperature')
parser.add_argument('--model_dir', default='output/', type=str, help='model save path')
parser.add_argument('--root', default="../datasets", type=str, help="optimizer")
parser.add_argument('--dataset', default='ImageNet', help='[ImageNet]')
parser.add_argument('--testset', default='ImageNet', help='[ImageNet, CIFAR10, CIFAR100, STL-10, cu_birds, dtd, traffic_sign, vgg_flower]')
args = parser.parse_args()

def train(loader, simclr_model, model, criterion, optimizer):
    loss_epoch = 0
    model.train()
    for step, (x, y) in enumerate(loader):
        optimizer.zero_grad()
        with torch.no_grad():
            if args.model == 'LBE':
                _, _, h, _, _, _, _ = simclr_model(x.cuda())
            elif args.model == 'MIB':
                h, _, _ = simclr_model(x.cuda())
            else:
                h, _ = simclr_model(x.cuda())
        output = model(h)
        loss = criterion(output, y.cuda())
        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()
        if step % 50 == 0:
            print(f"Step [{step}/{len(loader)}]\t Loss: {loss.item()}")
    return loss_epoch

def test(loader, simclr_model, model):
    right_num = 0
    all_num = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.cuda(), y.cuda()
            if args.model == 'LBE':
                _, _, h, _, _, _, _ = simclr_model(x)
            elif args.model == 'MIB':
                h, _, _ = simclr_model(x)
            else:
                h, _ = simclr_model(x)
            output = model(h)

            predicted = output.argmax(1)
            right_num += (predicted == y).sum().item()
            all_num += y.size(0)
    accuracy = right_num*100./all_num
    return accuracy

def load_transform(dataset, split='train'):
    mean, std = get_data_mean_and_stdev(dataset)
    if split=='train':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
    return transform

def main():
    data = 'imagenet'
    if args.testset == "CIFAR10":
        train_dataset = torchvision.datasets.CIFAR10(args.root, train=True, download=True, transform=load_transform('CIFAR10', 'train'))
        test_dataset = torchvision.datasets.CIFAR10(args.root, train=False, download=True, transform=load_transform('CIFAR10', 'test'))
    elif args.testset == "CIFAR100":
        train_dataset = torchvision.datasets.CIFAR100(args.root, train=True, download=True, transform=load_transform('CIFAR100', 'train'))
        test_dataset = torchvision.datasets.CIFAR100(args.root, train=False, download=True, transform=load_transform('CIFAR100', 'test'))
    elif args.testset == "STL-10":
        train_dataset = torchvision.datasets.STL10(args.root, split='train', download=True, transform=load_transform('STL-10', 'train'))
        test_dataset = torchvision.datasets.STL10(args.root, split='test', download=True, transform=load_transform('STL-10', 'test'))
    elif args.testset == "ImageNet":
        train_dataset = torchvision.datasets.ImageFolder(os.path.join(args.root, 'ImageNet/train'), load_transform('ImageNet', 'train'))
        test_dataset = torchvision.datasets.ImageFolder(os.path.join(args.root, 'ImageNet/val'), load_transform('ImageNet', 'test'))
    else:
        train_dataset = DATASET[args.testset](train=True, image_transforms=load_transform(args.testset, 'train'))
        test_dataset = DATASET[args.testset](train=False, image_transforms=load_transform(args.testset, 'test'))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.logistic_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.logistic_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers)

    log_dir = args.model_dir + "log/" + args.testset + '_%s/'%args.model
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    suffix = args.dataset + '_{}_batch_{}'.format(args.resnet, args.batch_size)
    suffix = suffix + '_proj_dim_{}'.format(args.projection_dim) + '_epoch_%d'%args.epochs
    args.model_dir = args.model_dir + 'checkpoint/' + args.dataset + '_%s/'%args.model
    epoch_dir = args.model_dir + suffix + '.pt'
    print("Loading {}".format(epoch_dir))
    simclr_model, _, _, _ = load_model(args, reload_model=True, load_path=epoch_dir, data=data)
    simclr_model = simclr_model.cuda()
    simclr_model.eval()

    # Logistic Regression
    n_classes = get_data_nclass(args.testset)
    model = LogisticRegression(simclr_model.n_features, n_classes).cuda()
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.logistic_epochs)
    criterion = torch.nn.CrossEntropyLoss()

    best_acc = 0.
    test_log_file = open(log_dir + suffix + '_LR.txt', "w")
    for epoch in range(args.logistic_epochs):
        loss_epoch = train(train_loader, simclr_model, model, criterion, optimizer)
        print("Train Epoch [{}]\t Average loss: {}".format(epoch, loss_epoch/len(train_loader)))
        print("Train Epoch [{}]\t Average loss: {}".format(epoch, loss_epoch/len(train_loader)), file=test_log_file)
        test_log_file.flush()

        # final testing
        test_current_acc = test(test_loader, simclr_model, model)
        if test_current_acc > best_acc:
            best_acc = test_current_acc
        print("Test Epoch [{}]\t Accuracy: {}\t Best Accuracy: {}\n".format(epoch, test_current_acc, best_acc))
        print("Test Epoch [{}]\t Accuracy: {}\t Best Accuracy: {}\n".format(epoch, test_current_acc, best_acc), file=test_log_file)
        test_log_file.flush()
        scheduler.step()

    print("Final Best Accuracy: {}".format(best_acc))
    print("Final Best Accuracy: {}".format(best_acc), file=test_log_file)
    test_log_file.flush()

if __name__ == "__main__":
    main()