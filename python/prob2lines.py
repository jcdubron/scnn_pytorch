import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image

def getLane(probmap, pts):
    thr = 0.3
    coordinate = np.zeros(pts);
    for i in range(pts):
        line = probmap[round(288-i*20/590*288)-1]
        if np.max(line)/255 > thr:
            coordinate[i] = np.argmax(line)+1
    if np.sum(coordinate>0) < 2:
        coordinate = np.zeros(pts)
    return coordinate

def prob2lines(prob_dir, out_dir, list_file):
    # parser = argparse.ArgumentParser(description='Transform probmap to lines')
    # parser.add_argument('--prob_dir', metavar='DIR', default='/home/dwt/scnn_pytorch',
                        # help='path to probmap (default: /home/dwt/scnn_pytorch)')
    # parser.add_argument('--out_dir', metavar='DIR', default='/home/dwt/scnn_pytorch',
                        # help='path to output (default: /home/dwt/scnn_pytorch)')
    # parser.add_argument('--list_file', metavar='DIR', default='eval.txt',
                        # help='path to probmap (default: eval.txt)')
    # args = parser.parse_args()
    
    lists = pd.read_csv(list_file, sep=' ', header=None,
                        names=('img', 'probmap', 'label1', 'label2', 'label3', 'label4'))
    pts = 18
    
    for k, im in enumerate(lists['img'], 1):
        existPath = prob_dir + im[:-4] + '.exist.txt'
        outname = out_dir + im[:-4] + '.lines.txt'
        prefix = '/'.join(outname.split('/')[:-1])
        if not os.path.exists(prefix):
            os.makedirs(prefix)
        f = open(outname, 'w')
        
        labels = list(pd.read_csv(existPath, sep=' ', header=None).iloc[0])
        coordinates = np.zeros((4, pts))
        for i in range(4):
            if labels[i] == 1:
                probfile = prob_dir + im[:-4] + '_{0}_avg.png'.format(i+1)
                probmap = np.array(Image.open(probfile))
                coordinates[i] = getLane(probmap, pts)

                if np.sum(coordinates[i]>0) > 1:
                    for idx, value in enumerate(coordinates[i]):
                        if value > 0:
                            f.write('%d %d ' % (round(value*1640/800)-1, round(590-idx*20)-1))
                    f.write('\n')
        f.close()
        ''''''''''''''''''''''''''''''
        if k % 100 == 0:
            print('{0} images have been processed from prob to lines.'.format(k))
        ''''''''''''''''''''''''''''''
