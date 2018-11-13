import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

class LaneDataset(Dataset):
    def __init__(self, img_dir, prob_dir, list_file, tag=False, transform=None):
        self.img_dir = img_dir
        self.prob_dir = prob_dir
        self.transform = transform
        self.tag = tag
        self.data_list = pd.read_csv(list_file, sep=' ', header=None,
                                     names=('image', 'probmap', 'label1', 'label2', 'label3', 'label4'))
    
    def __len__(self):
        return self.data_list.shape[0]
    
    def __getitem__(self, idx):
        img_name = self.img_dir + self.data_list.iloc[idx, 0]
        image = Image.open(img_name)
        probmap_name = self.prob_dir + self.data_list.iloc[idx, 1]
        probmap = Image.open(probmap_name)
        labels = torch.tensor(self.data_list.iloc[idx, 2:6], dtype=torch.double)
        
        if self.tag:
            sample = {'image': image, 'probmap': probmap, 'labels': labels, 'file': self.data_list.iloc[idx, 0][:-4]}
        else:
            sample = {'image': image, 'probmap': probmap, 'labels': labels}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
