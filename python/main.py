import os
import time
import cv2
import pandas as pd
import torch
import argparse
import subprocess
import numpy as np
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter

import sys
sys.path.append('.')
from python.model import SCNN
from python.transforms import *
from python.dataset import LaneDataset
from tools.prob2lines import *

def init_weights(m):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        torch.nn.init.xavier_normal_(m, gain=1)
    elif 'BatchNorm' in classname:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train(model, writer, args, device, train_loader, eval_loader, scheduler, epoch_start, loss_weight=(1, 1)):
    # set training mode
    print('Start training.')
    model.train()
    # eval(model, writer, device, test_loader)
    
    batch_num = 0
    avg_loss = 0
    for epoch in range(epoch_start, args.epoches):
        end = time.time()
        
        # loss sum during one epoch
        spacial_loss = 0
        label_loss = 0
        
        # iterate over one epoch
        for batch_idx, sample in enumerate(train_loader, 1):
            # load data
            data = sample['image'].to(device)
            probmap = sample['probmap'].to(device)
            label = sample['labels'].to(device)
            
            # forward / backward / optimize
            scheduler.optimizer.zero_grad()
            predictmap, predict = model(data)
            weight = torch.tensor([0.4, 1, 1, 1, 1], dtype=torch.double).to(device)
            loss1 = F.cross_entropy(predictmap, probmap, weight=weight)
            loss2 = F.binary_cross_entropy(predict, label)
            loss = loss1 * loss_weight[0] + loss2 * loss_weight[1]
            loss.backward()
            scheduler.step()
            
            # update loss sum
            spacial_loss += loss1
            label_loss += loss2
            batch_num += 1
            avg_loss = avg_loss * 0.99 + loss * 0.01
            
            if batch_idx % args.log_interval == 0:
                spacial_loss /= args.log_interval
                label_loss /= args.log_interval
                total_loss = spacial_loss * loss_weight[0] + label_loss * loss_weight[1]
                writer.add_scalar('lr', scheduler.optimizer.param_groups[0]['lr'], batch_num)
                writer.add_scalar('train/spacial_loss', spacial_loss, batch_num)
                writer.add_scalar('train/label_loss', label_loss, batch_num)
                writer.add_scalar('train/total_loss', total_loss, batch_num)
                writer.add_scalar('train/avg_loss', avg_loss/(1-np.power(0.99, batch_num)))
                batch_time = time.time() - end
                print('Epoch: [{0}][{1}/{2}]  '
                      'Time: {batch_time:.3f}  '
                      'SpacialLoss: {spacial_loss:.4f}  '
                      'LabelLoss: {label_loss:.4f}  '
                      'TotalLoss: {loss:.4f}  '
                      'AverageLoss: {avg_loss:.4f}'.format(
                      epoch, batch_idx, len(train_loader),
                      batch_time=batch_time, spacial_loss=spacial_loss,
                      label_loss=label_loss, loss=total_loss, avg_loss=avg_loss/(1-np.power(0.99, batch_num))))
                spacial_loss = 0
                label_loss = 0
                end = time.time()
        
        if epoch % args.snapshot_interval == 0:
            torch.save({'model_state_dict': model.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'epoch': epoch}, '{0}_epoch_{1}.pth'.format(args.snapshot_prefix, epoch))
            print('The snapshot is saved in {0}_epoch_{1}.pth'.format(args.snapshot_prefix, epoch))
            # torch.save(model.state_dict(), '{0}_{1}.pth'.format(args.snapshot_prefix, epoch))
            # torch.save(model, args.snapshot_prefix + str(batch_count) + '.pth')
        
        eval(model, device, args, eval_loader, writer=writer, epoch=epoch, loss_weight=loss_weight)


def eval(model, device, args, eval_loader, writer=None, epoch=None, loss_weight=(1, 1)):
    print('Start evaluate the model on eval dataset.')
    model.eval()
    eval_spacial_loss = 0
    eval_label_loss = 0
    with torch.no_grad():
        for sample in eval_loader:
            data = sample['image'].to(device)
            probmaps = sample['probmap'].to(device)
            labels = sample['labels'].to(device)
            predictmaps, predicts = model(data)
            weight = torch.tensor([0.4, 1, 1, 1, 1], dtype=torch.double).to(device)
            eval_spacial_loss += F.cross_entropy(predictmaps, probmaps, weight=weight)
            eval_label_loss += F.binary_cross_entropy(predicts, labels)

            for i in range(args.batch_size):
                file_name = './output/probmap' + sample['file'][i]
                prefix = '/'.join(file_name.split('/')[:-1])
                if not os.path.exists(prefix):
                    os.makedirs(prefix)
                
                predict = predicts[i]
                predict = 1*(np.array(predict.cpu())>0)
                f = open(file_name+'.exist.txt', 'w')
                f.write('{0} {1} {2} {3}'.format(predict[0], predict[1], predict[2], predict[3]))
                f.close()
                
                predictmap = predictmaps[i, ...]
                predictmap = np.array(predictmap.cpu())*255
                predictmap = predictmap*(predictmap==np.max(predictmap, axis=0))
                predictmap = np.transpose(predictmap, (1,2,0))
                for j in range(3):
                    predictmap = cv2.blur(predictmap, (3,3), borderType=cv2.BORDER_CONSTANT)
                predictmap = np.transpose(predictmap, (2,0,1))
                
                for j in range(1, 5):
                    cv2.imwrite(file_name+'_{0}_avg.png'.format(j), predictmap[j])
    
    prob2lines('./output/probmap', './output/lines', args.eval_list_file)
    result = subprocess.check_output('./tools/evaluate -a {0} -d ./output/lines/'
                                     '-i {1} -l {2} -w 30 -t 0.5 -c 1640 -r 590 -f 1'.format(
                                     args.data_dir, args.data_dir, args.eval_list_file), shell=True)
    eval_spacial_loss /= len(eval_loader.dataset)
    eval_label_loss /= len(eval_loader.dataset)
    eval_total_loss = eval_spacial_loss*loss_weight[0] + eval_label_loss*loss_weight[1]
    if writer is not None:
        writer.add_scalar('eval/spacial_loss', eval_spacial_loss, epoch)
        writer.add_scalar('eval/label_loss', eval_label_loss, epoch)
        writer.add_scalar('eval/total_loss', eval_total_loss, epoch)
    print('[Eval results] Spacial loss: {:.4f}, Label loss: {:.4f}, Total loss: {:.4f},'.format(
          eval_spacial_loss, eval_label_loss, eval_total_loss))
    print('[Eval results]', result.decode('utf-8').strip())

def main():
    # parser
    parser = argparse.ArgumentParser(description='PyTorch SCNN Model')
    parser.add_argument('--data_dir', metavar='DIR', default='/home/dwt/scnn_pytorch',
                        help='path to dataset (default: /home/dwt/scnn_pytorch)')
    parser.add_argument('--train_list_file', metavar='DIR', default='train.txt',
                        help='train list file (default: train.txt)')
    parser.add_argument('--eval_list_file', metavar='DIR', default='eval.txt',
                        help='eval list file (default: eval.txt)')
    parser.add_argument('--batch_size', type=int, default=4, metavar='N',
                        help='batch size for training (default: 4)')
    parser.add_argument('--epoches', type=int, default=10, metavar='N',
                        help='number of epoches to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', default=1e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)')
    # parser.add_argument('--gpu', action='store_true', default=False,
    #                     help='GPU training')
    parser.add_argument('--gpu', metavar='N', type=int, nargs='+', default=0,
                        help='GPU ids')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--checkpoint', metavar='DIR',
                        help='use pre-trained model')
    parser.add_argument('--weights', metavar='DIR',
                        help='use finetuned model')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status (default: 100)')
    parser.add_argument('--snapshot_interval', type=int, default=1, metavar='N',
                        help='how many epoches to wait before saving snapshot (default: 2)')
    parser.add_argument('--snapshot_prefix', type=str, default='./snapshot/model', metavar='PATH',
                        help='snapshot prefix (default: ./snapshot/model)')
    parser.add_argument('--tensorboard', type=str, default='log', metavar='PATH',
                        help='tensorboard log path  (default: log)')
    args = parser.parse_args()
    
    # tensorboardX
    writer = SummaryWriter(args.tensorboard)
    
    # cuda and seed
    use_cuda = args.gpu and torch.cuda.is_available()
    device = torch.device('cuda:{0}'.format(args.gpu[0]) if use_cuda else 'cpu')
    torch.manual_seed(args.seed)
    if use_cuda:
        print('Use Device: GPU', args.gpu)
    else:
        print('Use Device: CPU')
    
    # dataset
    print('Start dataset loading initialization.')
    train_dataset = LaneDataset(img_dir=args.data_dir+'/train', prob_dir=args.data_dir+'/train_labelmap',
                                list_file=args.train_list_file, tag=False,
                                transform=transforms.Compose([SampleResize((800, 288)),
                                                              SampleRandomHFlip(0.5),
                                                              SampleRandomVFlip(0.5),
                                                              SampleToTensor(),
                                                              SampleNormalize(mean=[0.37042467, 0.36758537, 0.3584016],
                                                                              std=[0.5, 0.5, 0.5])]))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, drop_last=False)
    
    eval_dataset = LaneDataset(img_dir=args.data_dir+'/train', prob_dir=args.data_dir+'/train_labelmap',
                               list_file=args.eval_list_file, tag=True,
                               transform=transforms.Compose([SampleResize((800, 288)),
                                                             SampleToTensor(),
                                                             SampleNormalize(mean=[0.37042467, 0.36758537, 0.3584016],
                                                                             std=[0.5, 0.5, 0.5])]))
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, drop_last=False)
    print('Dataset loading initialization done.')
    
    # model and scheduler
    model = SCNN().to(device)
    if len(args.gpu) > 1:
        model = torch.nn.DataParallel(model, device_ids=args.gpu)
    # model.apply(init_weights)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.95)
    epoch_start = 1
    
    # continue training from checkpoint
    if args.checkpoint:
        assert os.path.isfile(args.checkpoint)
        print('Start loading checkpoint.')
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch_start = checkpoint['epoch']
        print('Loading checkpoint done.')
    # finetune training with weights
    elif args.weights:
        assert os.path.isfile(args.weights)
        print('Start loading weights.')
        model_dict = model.state_dict()
        weights = torch.load(args.weights)
        weights = {k: v for k, v in weights.items() if k in model.state_dict()}
        model_dict.update(weights)
        model.load_state_dict(model_dict)
        print('Loading weights done.')
    
    model.double()
    # for sample in train_dataset:
    #     print(sample)
    # for idx, sample in enumerate(train_loader):
    #     print(idx, sample['image'].size(), sample['probmap'].size())
    #     input()
    # train
    train(model, writer, args, device, train_loader, eval_loader, scheduler, epoch_start, loss_weight=(1, 5000))


if __name__ == '__main__':
    main()
