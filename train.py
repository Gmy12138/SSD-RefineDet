from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from models.ssd import build_ssd
from eval import evaluate
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument("--epochs", type=int, default=1000, help="number of epochs")
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],type=str, help='VOC or COCO')
parser.add_argument('--dataset_root', default=VOC_ROOT, help='Dataset root directory path')
parser.add_argument('--basenet', default='weights/vgg16_reducedfc.pth', help='Pretrained base model')
parser.add_argument('--batch_size', default=8, type=int,help='Batch size for training')
parser.add_argument('--resume', default=None, type=str, help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,help='Gamma update for SGD')
parser.add_argument('--visdom', default=False, type=str2bool,help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='checkpoints/',help='Directory for saving checkpoint models')
parser.add_argument("--evaluation_interval", type=int, default=10, help="interval evaluations on validation set")
parser.add_argument('--save_folders', default='eval/', type=str,help='File path to save results')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def train():
    best_map = 0

    if args.dataset == 'VOC':
        if args.dataset_root == COCO_ROOT:
            parser.error('Must specify dataset if specifying dataset_root')
        cfg = voc
        dataset = VOCDetection(root=args.dataset_root, transform=SSDAugmentation(cfg['min_dim'], MEANS))



    net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])

    if args.cuda:
        # net = torch.nn.DataParallel(ssd_net)
        net = net.to(device)
        cudnn.benchmark = True

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        net.load_weights(args.resume)
    else:
        vgg_weights = torch.load(args.basenet)
        print('Loading base network...')
        net.vgg.load_state_dict(vgg_weights)
    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        net.extras.apply(weights_init)
        net.loc.apply(weights_init)
        net.conf.apply(weights_init)




    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5, False, args.cuda)

    # loss counters
    loc_loss = 0
    conf_loss = 0
    print('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size
    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)
    print(epoch_size)

    step_index = 0

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    # create batch iterator
    iteration=0

    for epoch in range(args.epochs):
        net.phase='train'
        net.train()

        for batch_i, ( images, targets) in enumerate(data_loader):

            iteration+=1

            if iteration in cfg['lr_steps']:
                step_index += 1
                adjust_learning_rate(optimizer, args.gamma, step_index)


            images = images.to(device)
            targets = [ann.to(device) for ann in targets]

            # forward
            t0 = time.time()
            out = net(images)
            # backprop
            optimizer.zero_grad()
            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c
            loss.backward()
            optimizer.step()
            t1 = time.time()

            loc_loss += loss_l.item()
            conf_loss += loss_c.item()

            if iteration % 10 == 0:
                print('timer: %.4f sec.' % (t1 - t0))
                print('epoch '+repr(epoch+1)+ ': iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.item()), end=' ')



        if epoch % args.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            map=evaluate(model=net,
                         save_folder=args.save_folders,
                         cuda=args.cuda,
                         top_k=5,
                         im_size=300,
                         thresh=0.05,
                         dataset_mean=((104, 117, 123)))

            if map >best_map:#取最好的map保存
                torch.save(net.state_dict(), f"checkpoints/ssd300_%d.pth" % (epoch+1))
                best_map = map




def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()




if __name__ == '__main__':
    train()
