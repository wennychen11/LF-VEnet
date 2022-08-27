import argparse
import math
import os
import sys
import time
import random
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader

from make_train_data import trainDataSet
from model_VEnet import VEnet

# Training settings
parser = argparse.ArgumentParser(description="LF-VEnet")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--step", type=int, default=2000, help="Learning rate decay every n epochs")
parser.add_argument("--reduce", type=float, default=0.5, help="Learning rate decay")
parser.add_argument("--patch_size", type=int, default=64, help="Training patch size")
parser.add_argument("--batch_size", type=int, default=1, help="Training batch size")
parser.add_argument("--resume_epoch", type=int, default=0, help="resume from checkpoint epoch")
parser.add_argument("--epoch_num", type=int, default=10000, help="maximum epoch for training")
parser.add_argument("--num_cp", type=int, default=20, help="Number of epochs for saving checkpoint")
parser.add_argument("--dataset_path", type=str, default="D:/datasets/LF-VEnet_training_data2.h5",
                    help="Dataset file for training")
parser.add_argument("--save_path", type=str, default="D:/datasets/VEnet_models/model_1.0/",
                    help="save path of model and training detail")
parser.add_argument("--n_view", type=int, default=7, help="Size of angular dim for training")
parser.add_argument("--scale", type=int, default=4, help="SR factor")
parser.add_argument('--gpu_no', type=int, default=1, help='GPU used: (default: %(default)s)')
parser.add_argument('--network_name', type=str, default='model_VEnet', help='name of training network')
parser.add_argument('--data_type', type=int, default=3, help="1: rresized, 2: lr_2, 3:lr_4")
parser.add_argument('--layer_num', type=int, default=10, help='layer number of network')
parser.add_argument('--random_seed', type=int, default=1, help='random seed of network')
opt = parser.parse_args()


class Logger:
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv3d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)


def get_loader(opt, shuffle=True, num_workers=0, pin_memory=True):
    train_set = trainDataSet(opt.dataset_path, patch_size=opt.patch_size, data_type=opt.data_type,
                             scale=opt.scale, n_view=opt.n_view)
    generator = torch.Generator()
    generator.manual_seed(opt.random_seed)
    data_loader = DataLoader(dataset=train_set,
                             batch_size=opt.batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers,
                             pin_memory=pin_memory,
                             generator=generator)
    return data_loader


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_no)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # rand seed
    torch.manual_seed(opt.random_seed)
    torch.cuda.manual_seed(opt.random_seed)
    np.random.seed(opt.random_seed)
    random.seed(opt.random_seed)
    print('create save and model directory...')
    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)
    sys.stdout = Logger(opt.save_path + 'train_{}_{}.log'.format(opt.network_name, int(time.time())), sys.stdout)

    print('training parameters:...')
    print(opt)

    print('loading datasets...')
    train_loader = get_loader(opt)
    print('loaded {} LFIs from {}'.format(len(train_loader), opt.dataset_path))

    print('using network {}'.format(opt.network_name))
    model = VEnet(n_view=opt.n_view, scale=opt.scale, layer_num=opt.layer_num).to(device)
    model.apply(weights_init_xavier)
    # network setting
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.step, gamma=opt.reduce)
    losslogger = defaultdict(list)

    if opt.resume_epoch:
        resume_path = os.path.join(opt.save_path, 'model_epoch_{}.pth'.format(opt.resume_epoch))
        if os.path.isfile(resume_path):
            print("loading checkpoint 'epoch{}'".format(resume_path))
            checkpoint = torch.load(resume_path)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            losslogger = checkpoint['losslogger']
        else:
            print("no model found at 'epoch{}'".format(opt.resume_epoch))

    print('training...')
    for epoch in range(opt.resume_epoch + 1, opt.epoch_num + 1):
        model.train()
        loss_count = 0.
        loss_count_s = 0.
        loss_count_e = 0.
        for i, batch in enumerate(train_loader, 1):
            lf_y = batch[0].to(device)
            lab = batch[1].to(device)
            sr = model(lf_y)
            s_loss = L1_Charbonnier_loss(sr, lab)
            e_loss = epi_loss(sr, lab)
            loss = s_loss + e_loss
            loss_count += loss.item()
            loss_count_s += s_loss.item()
            loss_count_e += e_loss.item()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.4)
            optimizer.step()

        scheduler.step()
        losslogger['epoch'].append(epoch)
        losslogger['loss'].append(loss_count / len(train_loader))
        losslogger['s_loss'].append(loss_count_s / len(train_loader))
        losslogger['e_loss'].append(loss_count_e / len(train_loader))
        print('[{}/{}] loss:{} s_loss:{} e_loss:{} lr:{}'.format(epoch, opt.epoch_num, loss_count / len(train_loader),
                                                                 loss_count_s / len(train_loader),
                                                                 loss_count_e / len(train_loader),
                                                                 optimizer.param_groups[0]['lr']))

        if epoch % opt.num_cp == 0:
            model_save_path = os.path.join(opt.save_path, "model_epoch_{}.pth".format(epoch))
            state = {'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'scheduler': scheduler.state_dict(), 'losslogger': losslogger}
            torch.save(state, model_save_path)
            print("checkpoint saved to {}".format(model_save_path))


def L1_Charbonnier_loss(X, Y):
    eps = 1e-6
    diff = torch.add(X, -Y)
    error = torch.sqrt(diff * diff + eps)
    loss = torch.sum(error) / torch.numel(error)
    return loss


def epi_loss(pred, label):
    def gradient(pred):
        D_dy = pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]
        D_dx = pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]
        D_day = pred[:, 1:, :, :, :] - pred[:, :-1, :, :, :]
        D_dax = pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]
        return D_dx, D_dy, D_dax, D_day

    N, an2, h, w = pred.shape
    an = int(math.sqrt(an2))
    pred = pred.view(N, an, an, h, w)
    label = label.view(N, an, an, h, w)

    pred_dx, pred_dy, pred_dax, pred_day = gradient(pred)
    label_dx, label_dy, label_dax, label_day = gradient(label)

    return L1_Charbonnier_loss(pred_dx, label_dx) + L1_Charbonnier_loss(pred_dy, label_dy) + L1_Charbonnier_loss(
        pred_dax, label_dax) + L1_Charbonnier_loss(pred_day, label_day)


if __name__ == '__main__':
    main()