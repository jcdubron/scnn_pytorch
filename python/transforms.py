import torch
import collections
import numpy as np
from PIL import Image


class SampleResize(object):
    def __init__(self, size):
        assert (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
    
    def __call__(self, sample):
        sample['image'] = sample['image'].resize(self.size, Image.BILINEAR)
        sample['probmap'] = sample['probmap'].resize(self.size, Image.NEAREST)
        return sample


class SampleRandomHFlip(object):
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, sample):
        if not isinstance(sample['image'], Image.Image):
            raise TypeError('img should be PIL Image. Got {}'.format(type(sample['image'])))
        if np.random.random() < self.p:
            sample['image'] = sample['image'].transpose(Image.FLIP_LEFT_RIGHT)
            sample['probmap'] = sample['probmap'].transpose(Image.FLIP_LEFT_RIGHT)
        return sample


class SampleRandomVFlip(object):
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, sample):
        if not isinstance(sample['image'], Image.Image):
            raise TypeError('img should be PIL Image. Got {}'.format(type(sample['image'])))
        if np.random.random() < self.p:
            sample['image'] = sample['image'].transpose(Image.FLIP_TOP_BOTTOM)
            sample['probmap'] = sample['probmap'].transpose(Image.FLIP_TOP_BOTTOM)
        return sample


class SampleToTensor(object):
    def __call__(self, sample):
        image = np.array(sample['image']).transpose((2, 0, 1)) / 255
        sample['image'] = torch.from_numpy(image).to(torch.float)
        probmap = np.array(sample['probmap'])
        sample['probmap'] = torch.from_numpy(probmap).to(torch.long)
        return sample


class SampleNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, sample):
        for t, m, s in zip(sample['image'], self.mean, self.std):
            t.sub_(m).div_(s)
        return sample


class TestSampleResize(object):
    def __init__(self, size):
        assert (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
    
    def __call__(self, sample):
        sample['image'] = sample['image'].resize(self.size, Image.BILINEAR)
        return sample


class TestSampleToTensor(object):
    def __call__(self, sample):
        image = np.array(sample['image']).transpose((2, 0, 1)) / 255
        sample['image'] = torch.from_numpy(image).to(torch.float)
        return sample


class TestSampleNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, sample):
        for t, m, s in zip(sample['image'], self.mean, self.std):
            t.sub_(m).div_(s)
        return sample
