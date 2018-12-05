import cv2
import torch
import subprocess
from .prob2lines import *
from .crf import *


def test(model, device, test_loader, args):
    print('Start testing.')
    model.eval()
    with torch.no_grad():
        index = 0
        for sample in test_loader:
            data = sample['image'].to(device)
            end = time.time()
            predictmaps, predicts = model(data)
            batch_time = time.time() - end
            
            for i in range(data.shape[0]):
                index += 1
                file_name = './output/probmap' + sample['file'][i]
                prefix = '/'.join(file_name.split('/')[:-1])
                if not os.path.exists(prefix):
                    os.makedirs(prefix)
                
                predict = predicts[i]
                predict = 1 * (np.array(predict.cpu()) > 0.5)
                f = open(file_name + '.exist.txt', 'w')
                f.write('{0} {1} {2} {3}'.format(predict[0], predict[1], predict[2], predict[3]))
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
                
                predictmap = np.array(predictmap.cpu()) * 255
                predictmap = predictmap * (predictmap == np.max(predictmap, axis=0))
                print(index, batch_time, predict, np.sum(predictmap[0] > 0), np.sum(predictmap[1] > 0),
                      np.sum(predictmap[2] > 0), np.sum(predictmap[3] > 0), np.sum(predictmap[4] > 0))
                predictmap = np.transpose(predictmap, (1, 2, 0))
                for j in range(3):
                    predictmap = cv2.blur(predictmap, (9, 9), borderType=cv2.BORDER_CONSTANT)
                predictmap = np.transpose(predictmap, (2, 0, 1))
                
                for j in range(1, 5):
                    cv2.imwrite(file_name + '_{0}_avg.png'.format(j), predictmap[j])
    
    prob2lines('./output/probmap', './output/lines', args.test_list_file)
    result = subprocess.check_output('./tools/evaluate -a {0} -d ./output/lines/ '
                                     '-i {1} -l {2} -w 30 -t 0.5 -c 1640 -r 590 -f 1'.format(
        args.test_data_dir, args.test_data_dir, args.test_list_file), shell=True)
    print('[Test results]', result.decode('utf-8').strip())
