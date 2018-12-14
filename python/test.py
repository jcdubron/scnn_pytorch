import cv2
import time
import torch
import subprocess
import torch.nn as nn
import torch.nn.functional as F
from .prob2lines import *

def test(model, device, test_loader, args):
    print('Start testing.')
    model.eval()
    with torch.no_grad():
        index = 0
        blur = nn.Conv2d(5, 5, 9, padding=4, bias=False, groups=5).to(device)
        nn.init.constant_(blur.weight, 1/81)
        
        for sample in test_loader:
            data = sample['image'].to(device)
            end = time.time()
            predictmaps, predicts = model(data)
            predictmaps = F.softmax(predictmaps, dim=1)
            batch_time = time.time() - end
            
            for i in range(data.shape[0]):
                index += 1
                file_name = './output/probmap' + sample['file'][i]
                prefix = '/'.join(file_name.split('/')[:-1])
                if not os.path.exists(prefix):
                    os.makedirs(prefix)
                
                predict = predicts[i] > 0.5
                # predict = 1 * (np.array(predict.cpu()) > 0.5)
                f = open(file_name + '.exist.txt', 'w')
                f.write('{:d} {:d} {:d} {:d}'.format(predict[0], predict[1], predict[2], predict[3]))
                f.close()
                
                predictmap = predictmaps[i, ...]
                # print('MSMF start.')
                # msmf = MultiStageMeanField(160, 3, 3, 3, [3, 5], [1, 5, 288, 800]).cuda()
                # msmf_output = msmf(predictmap.unsqueeze(0), predictmap.unsqueeze(0), data[i:i+1, ...]).squeeze()
                # for j in range(1, 5):
                #     cv2.imwrite('before_{0}.png'.format(j), np.array(predictmap[j].cpu()) * 255)
                #     cv2.imwrite('after_{0}.png'.format(j), np.array(msmf_output[j].cpu()) * 255)
                #
                # print('MSMF done.')
                # input()

                # probmaps = sample['probmap'].to(device)
                # labels = sample['labels'].to(device)
                # weight = torch.tensor([0.4, 1, 1, 1, 1], dtype=torch.double).to(device)
                # loss1 = F.cross_entropy(predictmaps, probmaps, weight=weight)
                # loss2 = F.binary_cross_entropy(predicts, labels)
                # print('{:.4f} {:.4f}'.format(loss1, loss2))
                
                predictmap = predictmap * 255
                predictmap = predictmap * (predictmap == torch.max(predictmap, dim=0)[0]).to(torch.float)
                print('{:d} {:.4f} {:.4f} {:d} {:d} {:d} {:d}'.format(
                      index, batch_time, time.time() - end,
                      torch.sum(predictmap[1] > 0), torch.sum(predictmap[2] > 0),
                      torch.sum(predictmap[3] > 0), torch.sum(predictmap[4] > 0)))
                for j in range(3):
                    predictmap = blur(predictmap.unsqueeze(0)).squeeze()
                
                # predictmap = np.array(predictmap.cpu()) * 255
                # predictmap = predictmap * (predictmap == np.max(predictmap, axis=0))
                # print(index, batch_time, predict, np.sum(predictmap[0] > 0), np.sum(predictmap[1] > 0),
                #       np.sum(predictmap[2] > 0), np.sum(predictmap[3] > 0), np.sum(predictmap[4] > 0))
                # predictmap = np.transpose(predictmap, (1, 2, 0))
                # for j in range(3):
                #     predictmap = cv2.blur(predictmap, (9, 9), borderType=cv2.BORDER_CONSTANT)
                # predictmap = np.transpose(predictmap, (2, 0, 1))

                predictmap = np.array(predictmap.cpu())
                for j in range(1, 5):
                    cv2.imwrite(file_name + '_{0}_avg.png'.format(j), predictmap[j])
    
    print('[Probmap to lines]')
    prob2lines('./output/probmap', './output/lines', args.test_list_file)
    result = subprocess.check_output('./tools/evaluate -a {0} -d ./output/lines/ '
                                     '-i {1} -l {2} -w 30 -t 0.5 -c 1640 -r 590 -f 1'.format(
                                     args.test_data_dir, args.test_data_dir, args.test_list_file), shell=True)
    print('[Evaluate Result]')
    print(result.decode('utf-8').strip())
