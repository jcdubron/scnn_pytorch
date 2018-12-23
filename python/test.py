import cv2
import time
import torch
import subprocess
import torch.nn as nn
import torch.nn.functional as F
from .prob2lines import *

def test(model, device, test_loader, args, list_idx=0):
    print('Start testing [{0}].'.format(args.test_list_file[list_idx]))
    model.eval()
    with torch.no_grad():
        index = 0
        blur = nn.Conv2d(5, 5, 9, padding=4, bias=False, groups=5).to(device)
        nn.init.constant_(blur.weight, 1/81)
        
        for sample in test_loader:
            data = sample['image'].to(device)
            predictmaps, predicts = model(data)
            predictmaps = F.softmax(predictmaps, dim=1)
            
            for i in range(data.shape[0]):
                index += 1
                file_name = './output/probmap' + sample['file'][i]
                prefix = '/'.join(file_name.split('/')[:-1])
                if not os.path.exists(prefix):
                    os.makedirs(prefix)
                
                predict = predicts[i] > 0.5
                f = open(file_name + '.exist.txt', 'w')
                f.write('{:d} {:d} {:d} {:d}'.format(predict[0], predict[1], predict[2], predict[3]))
                f.close()
                
                predictmap = predictmaps[i, ...]
                predictmap = predictmap * 255
                # predictmap = predictmap * (predictmap == torch.max(predictmap, dim=0)[0]).to(torch.float)
                predictmap = blur(predictmap.unsqueeze(0)).squeeze()
                predictmap = np.array(predictmap.cpu())
                for j in range(1, 5):
                    cv2.imwrite(file_name + '_{0}_avg.png'.format(j), predictmap[j])
                
                if index % 100 == 0:
                    print('{0} images have been tested.'.format(index))
    
    print('[Probmap to lines]')
    prob2lines('./output/probmap', './output/lines', args.test_list_file[list_idx])
    result = subprocess.check_output('./tools/evaluate -a {0} -d ./output/lines/ '
                                     '-i {1} -l {2} -w 30 -t 0.5 -c 1640 -r 590 -f 1'.format(
                                     args.test_data_dir, args.test_data_dir, args.test_list_file[list_idx]), shell=True)
    print('[Evaluate Result]')
    print(result.decode('utf-8').strip())
    
    f = open(args.test_log, 'a')
    f.write(args.test_list_file[list_idx] + '\n')
    f.write(result.decode('utf-8').strip() + '\n\n')
    f.close()
