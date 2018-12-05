import time
import signal
import cv2
import torch
import subprocess
import torch.nn.functional as F
from .prob2lines import *


def train(model, writer, args, device, train_loader, eval_loader, scheduler, epoch_start, loss_weight=(1, 1)):
    # set training mode
    print('Start training.')
    model.train()
    # eval(model, writer, device, test_loader)
    
    batch_num = 0
    avg_loss = 0
    
    # Ctrl+C signal operation
    def SaveModel(signum, frame):
        torch.save({'model_state_dict': model.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'batch_num': batch_num}, '{0}_batch_{1}.pth'.format(args.snapshot_prefix, batch_num))
        print('The snapshot is saved in {0}_batch_{1}.pth'.format(args.snapshot_prefix, batch_num))
    signal.signal(signal.SIGINT, SaveModel)
    
    for epoch in range(epoch_start, args.epoches):
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
            end = time.time()
            predictmap, predict = model(data)
            batch_time = time.time() - end
            weight = torch.tensor([0.4, 1, 1, 1, 1], dtype=torch.double).to(device)
            loss1 = F.cross_entropy(predictmap, probmap, weight=weight)
            loss2 = F.binary_cross_entropy(predict, label)
            # print(predict, label, loss2, F.binary_cross_entropy(predict[0], label[0]),
            #       F.binary_cross_entropy(predict[1], label[1]), F.binary_cross_entropy(predict[2], label[2]),
            #       F.binary_cross_entropy(predict[3], label[3]))
            loss = loss1 * loss_weight[0] + loss2 * loss_weight[1]
            loss.backward()
            scheduler.optimizer.step()
            scheduler.step()
            
            # for i in range(4):
            #     for j in range(1, 5):
            #         cv2.imwrite('a{1}_{0}_avg.png'.format(i, j), predictmap[i, j].cpu().detach().numpy()*255)
            #         cv2.imwrite('b{1}_{0}_avg.png'.format(i, j), probmap[i].cpu().detach().numpy()*255)
            # input()
            
            # update loss sum
            # spacial_loss += loss1
            # label_loss += loss2
            # avg_loss = avg_loss * 0.99 + loss * 0.01
            batch_num += 1
            spacial_loss = spacial_loss * 0.99 + loss1 * 0.01
            label_loss = label_loss * 0.99 + loss2 * 0.01
            
            if batch_idx % args.log_interval == 0:
                # spacial_loss /= args.log_interval
                # label_loss /= args.log_interval
                total_loss = spacial_loss * loss_weight[0] + label_loss * loss_weight[1]
                writer.add_scalar('lr', scheduler.optimizer.param_groups[0]['lr'], batch_num)
                writer.add_scalar('train/spacial_loss', spacial_loss / (1 - np.power(0.99, batch_num)), batch_num)
                writer.add_scalar('train/label_loss', label_loss / (1 - np.power(0.99, batch_num)), batch_num)
                writer.add_scalar('train/total_loss', total_loss / (1 - np.power(0.99, batch_num)), batch_num)
                print('Epoch:[{0}][{1}/{2}]  '
                      'LR: {lr:.4f}  '
                      'Time: {batch_time:.3f}  '
                      'SpacialLoss: {spacial_loss:.4f}  '
                      'LabelLoss: {label_loss:.4f}  '
                      'TotalLoss: {loss:.4f}'.format(
                    epoch, batch_idx, len(train_loader), lr=scheduler.optimizer.param_groups[0]['lr'],
                    batch_time=batch_time, spacial_loss=spacial_loss / (1 - np.power(0.99, batch_num)),
                    label_loss=label_loss / (1 - np.power(0.99, batch_num)), loss=total_loss / (1 - np.power(0.99, batch_num))))
                spacial_loss = 0
                label_loss = 0
        
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
                predict = 1* (np.array(predict.cpu()) > 0.5)
                f = open(file_name + '.exist.txt', 'w')
                f.write('{0} {1} {2} {3}'.format(predict[0], predict[1], predict[2], predict[3]))
                f.close()
                
                predictmap = predictmaps[i, ...]
                predictmap = np.array(predictmap.cpu()) * 255
                predictmap = predictmap * (predictmap == np.max(predictmap, axis=0))
                # print(predict)
                # print(np.sum(predictmap[0] > 0), np.sum(predictmap[1] > 0), np.sum(predictmap[2] > 0), np.sum(predictmap[3] > 0), np.sum(predictmap[4] > 0),)
                predictmap = np.transpose(predictmap, (1, 2, 0))
                for j in range(3):
                    predictmap = cv2.blur(predictmap, (9, 9), borderType=cv2.BORDER_CONSTANT)
                predictmap = np.transpose(predictmap, (2, 0, 1))
                
                for j in range(1, 5):
                    cv2.imwrite(file_name + '_{0}_avg.png'.format(j), predictmap[j])
    
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
    print('[Eval results] Spacial loss: {:.4f}, Label loss: {:.4f}, Total loss: {:.4f},'.format(
        eval_spacial_loss, eval_label_loss, eval_total_loss))
    print('[Eval results]', result.decode('utf-8').strip())