import os
import torch
from modules import Pretrain, ReCon32

def load_model(args, nclass, reload_model=False, load_path=None):
    model = Pretrain(args, nclass)
    if reload_model:
        if os.path.isfile(load_path):
            model_fp = os.path.join(load_path)
        else:
            print("No file to load")
            return
        model.load_state_dict(torch.load(model_fp, map_location=lambda storage, loc: storage))
    model = model.cuda()

    if args.model == 'RC':
        recon = ReCon32(512).cuda()
        params = [{'params': model.parameters()}, {'params': recon.parameters()}]
    else:
        recon = None
        params = model.parameters()

    scheduler = None
    optimizer = torch.optim.Adam(params, lr=args.lr)
    return model, recon, optimizer, scheduler

def save_model(model_dir, model, epoch):
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), model_dir + '_epoch_{}.pt'.format(epoch))
    else:
        torch.save(model.state_dict(), model_dir + '_epoch_{}.pt'.format(epoch))