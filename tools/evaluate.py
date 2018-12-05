import os
import numpy as np

def read_lines_file(file):
    assert os.path.isfile(file)
    f = open(file)
    lines = []
    for line in f:
        data = line[:-2].split(' ')
        curr_line = np.empty((0, 2))
        for i in range(len(data)//2):
            new_line = np.array([[int(data[i*2], int(data[i*2+1]))]])
            curr_line = np.vstack((curr_line, new_line))
        lines.append(curr_line)
    return lines

def evaluate(gt_lines_dir, pd_lines_dir, list_file, threshold=0.5, width=30):
    assert os.path.isfile(list_file)
    assert threshold < 1 and threshold > 0
    assert width >= 1
    
    f = open(list_file)
    for img in f:
        img = img[:-5] + '.lines.txt'
        gt_lines = read_lines_file(gt_lines_dir + img)
        pd_lines = read_lines_file(pd_lines_dir + img)
        
        # ......
        
        return
            
        