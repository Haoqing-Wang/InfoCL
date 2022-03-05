import os
import torch
from modules import Online, Target, ReCon32, ReCon64, ReCon224, LARS

def load_model(args, reload_model=False, load_path=None, data='non_imagenet'):
    Online_net = Online(args, data=data)
    Target_net = Target(args, data=data)
    if reload_model:
        if os.path.isfile(load_path):
            model_fp = os.path.join(load_path)
        else:
            print("No file to load")
            return
        Online_net.load_state_dict(torch.load(model_fp, map_location=lambda storage, loc: storage))
    Online_net = Online_net.cuda()
    Target_net = Target_net.cuda()

    if args.model == 'RC':
        if args.dataset == 'STL-10':
            recon = ReCon64(512).cuda()
        elif args.dataset == 'ImageNet':
            recon = ReCon224(2048).cuda()
        else:
            recon = ReCon32(512).cuda()
        params = [{'params': Online_net.parameters()}, {'params': recon.parameters()}]
    else:
        recon = None
        params = Online_net.parameters()

    scheduler = None
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(params, lr=args.lr)
    elif args.optimizer == "LARS":
        # LearningRate=(0.3×BatchSize/256) and weight decay of 10−6.
        learning_rate = 0.3 * args.batch_size / 256
        optimizer = LARS(params, lr=learning_rate, weight_decay=args.weight_decay, exclude_from_weight_decay=["batch_normalization", "bias"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0, last_epoch=-1)
    else:
        raise NotImplementedError

    return Online_net, Target_net, recon, optimizer, scheduler


def save_model(model_dir, model, epoch):
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), model_dir + '_epoch_{}.pt'.format(epoch))
    else:
        torch.save(model.state_dict(), model_dir + '_epoch_{}.pt'.format(epoch))