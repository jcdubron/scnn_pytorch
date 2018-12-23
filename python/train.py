import time
import signal
import cv2
import torch
import subprocess
import torch.nn as nn
import torch.nn.functional as F
from .prob2lines import *


def train(model, writer, args, device, train_loader, eval_loader, scheduler, epoch_start, loss_weight=(1, 1)):
    # set training mode
    print('Start training.')
    model.train()
    batch_num = 0

    # Ctrl+C signal operation
    def SaveModel(signum, frame):
        torch.save({'model_state_dict': model.state_dict(),
                    # 'scheduler_state_dict': scheduler.state_dict(),
                    'batch_num': batch_num}, '{0}_batch_{1}.pth'.format(args.snapshot_prefix, batch_num))
        print('The snapshot is saved in {0}_batch_{1}.pth'.format(args.snapshot_prefix, batch_num))
    signal.signal(signal.SIGINT, SaveModel)
    
    spacial_loss = 0
    label_loss = 0

    for epoch in range(epoch_start, args.epoches + 1):
        weight = torch.tensor([args.bg_weight, 1, 1, 1, 1]).to(device)
        
        # iterate over one epoch
        for batch_idx, sample in enumerate(train_loader, 1):
            batch_num += 1
            end = time.time()
            
            # load data
            data = sample['image'].to(device)
            probmaps = sample['probmap'].to(device)
            labels = sample['labels'].to(device)
            
            # forward
            scheduler.optimizer.zero_grad()
            predictmaps, predicts = model(data)
            
            # loss
            loss1 = F.cross_entropy(predictmaps, probmaps, weight=weight)
            loss2 = F.binary_cross_entropy(predicts, labels)
            loss = loss1 * loss_weight[0] + loss2 * loss_weight[1]
            spacial_loss = spacial_loss * 0.99 + loss1 * 0.01
            label_loss = label_loss * 0.99 + loss2 * 0.01
            
            # backward
            loss.backward()
            scheduler.optimizer.step()
            scheduler.step()

            # logging
            batch_time = time.time() - end
            if batch_idx % args.log_interval == 0:
                total_loss = spacial_loss * loss_weight[0] + label_loss * loss_weight[1]
                writer.add_scalar('lr', scheduler.optimizer.param_groups[0]['lr'], batch_num)
                writer.add_scalar('train/spacial_loss', spacial_loss / (1 - np.power(0.99, batch_num)), batch_num)
                writer.add_scalar('train/label_loss', label_loss / (1 - np.power(0.99, batch_num)), batch_num)
                writer.add_scalar('train/total_loss', total_loss / (1 - np.power(0.99, batch_num)), batch_num)
                print('Epoch:[{0}][{1}/{2}]  '
                      'LR: {lr:.6f}  '
                      'Time: {batch_time:.4f}  '
                      'SpacialLoss: {loss1:.4f}({spacial_loss:.4f})  '
                      'LabelLoss: {loss2:.4f}({label_loss:.4f})  '
                      'TotalLoss: {loss:.4f}({total_loss:.4f})'.format(
                      epoch, batch_idx, len(train_loader), lr=scheduler.optimizer.param_groups[0]['lr'],
                      batch_time=batch_time, spacial_loss=spacial_loss / (1 - np.power(0.99, batch_num)), loss1=loss1,
                      label_loss=label_loss / (1 - np.power(0.99, batch_num)), loss2=loss2,
                      total_loss=total_loss / (1 - np.power(0.99, batch_num)), loss=loss))
            
            if batch_num == args.batches:
                SaveModel(None, None)
                print('Training done.')
                return

        # snapshot
        if epoch % args.snapshot_interval == 0:
            torch.save({'model_state_dict': model.state_dict(),
                        # 'scheduler_state_dict': scheduler.state_dict(),
                        'epoch': epoch}, '{0}_epoch_{1}.pth'.format(args.snapshot_prefix, epoch))
            print('The snapshot is saved in {0}_epoch_{1}.pth'.format(args.snapshot_prefix, epoch))
        
        # eval
        eval(model, device, args, eval_loader, writer=writer, epoch=epoch, loss_weight=loss_weight)


def eval(model, device, args, eval_loader, writer=None, epoch=None, loss_weight=(1, 1)):
    print('Start evaluate the model on eval dataset.')
    model.eval()
    eval_spacial_loss = 0
    eval_label_loss = 0
    
    with torch.no_grad():
        index = 0
        weight = torch.tensor([args.bg_weight, 1, 1, 1, 1]).to(device)
        blur = nn.Conv2d(5, 5, 9, padding=4, bias=False, groups=5).to(device)
        nn.init.constant_(blur.weight, 1 / 81)
        
        for sample in eval_loader:
            # load data
            data = sample['image'].to(device)
            probmaps = sample['probmap'].to(device)
            labels = sample['labels'].to(device)
            
            # forward
            predictmaps, predicts = model(data)
            
            # loss
            eval_spacial_loss += F.cross_entropy(predictmaps, probmaps, weight=weight)
            eval_label_loss += F.binary_cross_entropy(predicts, labels)
            
            predictmaps = F.softmax(predictmaps, dim=1)
            for i in range(data.shape[0]):
                index += 1
                file_name = './output/probmap' + sample['file'][i]
                prefix = '/'.join(file_name.split('/')[:-1])
                if not os.path.exists(prefix):
                    os.makedirs(prefix)
                
                # existence label
                predict = predicts[i] > 0.5
                f = open(file_name + '.exist.txt', 'w')
                f.write('{:d} {:d} {:d} {:d}'.format(predict[0], predict[1], predict[2], predict[3]))
                f.close()
                
                # predict probmap of each class
                predictmap = predictmaps[i, ...]
                predictmap = predictmap * 255
                # predictmap = predictmap * (predictmap == torch.max(predictmap, dim=0)[0]).to(torch.float)
                predictmap = blur(predictmap.unsqueeze(0)).squeeze()
                predictmap = np.array(predictmap.cpu())
                for j in range(1, 5):
                    cv2.imwrite(file_name + '_{0}_avg.png'.format(j), predictmap[j])
                
                if index % 100 == 0:
                    print('{0} images have been predicted.'.format(index))
                
    prob2lines('./output/probmap', './output/lines', args.eval_list_file)
    result = subprocess.check_output('./tools/evaluate -a {0} -d ./output/lines/ '
                                     '-i {1} -l {2} -w 30 -t 0.5 -c 1640 -r 590 -f 1'.format(
                                     args.eval_data_dir, args.eval_data_dir, args.eval_list_file), shell=True)
    eval_spacial_loss /= len(eval_loader.dataset)
    eval_label_loss /= len(eval_loader.dataset)
    eval_total_loss = eval_spacial_loss * loss_weight[0] + eval_label_loss * loss_weight[1]
    if writer is not None:
        writer.add_scalar('eval/spacial_loss', eval_spacial_loss, epoch)
        writer.add_scalar('eval/label_loss', eval_label_loss, epoch)
        writer.add_scalar('eval/total_loss', eval_total_loss, epoch)
    print('[Eval results]')
    print('Spacial loss: {:.4f}, Label loss: {:.4f}, Total loss: {:.4f},'.format(
          eval_spacial_loss, eval_label_loss, eval_total_loss))
    print('[Evaluate Result]')
    print(result.decode('utf-8').strip())
