import torch
from torch import nn
import torch.nn.functional as F


class SCNN(nn.Module):
    def __init__(self):
        super(SCNN, self).__init__()
        
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(64)
        
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1, bias=False)
        self.bn2_2 = nn.BatchNorm2d(128)
        
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1, bias=False)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1, bias=False)
        self.bn3_3 = nn.BatchNorm2d(256)
        
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1, bias=False)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1, bias=False)
        self.bn4_3 = nn.BatchNorm2d(512)
        
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=2, dilation=2, bias=False)
        self.bn5_1 = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=2, dilation=2, bias=False)
        self.bn5_2 = nn.BatchNorm2d(512)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=2, dilation=2, bias=False)
        self.bn5_3 = nn.BatchNorm2d(512)
        
        self.conv6 = nn.Conv2d(512, 1024, 3, padding=4, dilation=4, bias=False)
        self.bn6 = nn.BatchNorm2d(1024)
        
        self.conv7 = nn.Conv2d(1024, 128, 1, bias=False)
        self.bn7 = nn.BatchNorm2d(128)
        
        self.conv_d = nn.Conv2d(128, 128, (1, 5), padding=(0, 2))
        self.conv_u = nn.Conv2d(128, 128, (1, 5), padding=(0, 2))
        self.conv_r = nn.Conv2d(128, 128, (5, 1), padding=(2, 0))
        self.conv_l = nn.Conv2d(128, 128, (5, 1), padding=(2, 0))
        
        self.conv8 = nn.Conv2d(128, 5, 1)
        
        self.fc9 = nn.Linear(5*18*50, 128)
        self.fc10 = nn.Linear(128, 4)
    
    def forward(self, x):
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = F.max_pool2d(x, 2, stride=2, padding=0)
        
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = F.max_pool2d(x, 2, stride=2, padding=0)
        
        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = F.relu(self.bn3_3(self.conv3_3(x)))
        x = F.max_pool2d(x, 2, stride=2, padding=0)
        
        x = F.relu(self.bn4_1(self.conv4_1(x)))
        x = F.relu(self.bn4_2(self.conv4_2(x)))
        x = F.relu(self.bn4_3(self.conv4_3(x)))
        
        x = F.relu(self.bn5_1(self.conv5_1(x)))
        x = F.relu(self.bn5_2(self.conv5_2(x)))
        x = F.relu(self.bn5_3(self.conv5_3(x)))
        
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        
        for i in range(1, x.shape[2]):
            x[..., i:i+1, :].add_(F.relu(self.conv_d(x[..., i-1:i, :])))
        
        for i in range(x.shape[2] - 2, 0, -1):
            x[..., i:i+1, :].add_(F.relu(self.conv_u(x[..., i+1:i+2, :])))
        
        for i in range(1, x.shape[3]):
            x[..., i:i+1].add_(F.relu(self.conv_r(x[..., i-1:i])))
        
        for i in range(x.shape[3] - 2, 0, -1):
            x[..., i:i+1].add_(F.relu(self.conv_l(x[..., i+1:i+2])))
        
        x = F.dropout2d(x, p=0.1)
        
        x = self.conv8(x)
        x1 = x.clone()
        x2 = x.clone()
        
        x1 = F.interpolate(x1, size=[288, 800], mode='bilinear', align_corners=True)
        x1 = F.softmax(x1, dim=1)
        x2 = F.softmax(x2, dim=1)
        x2 = F.avg_pool2d(x2, 2, stride=2, padding=0)
        x2 = x2.view(-1, x2.numel() // x2.shape[0])
        x2 = self.fc9(x2)
        x2 = F.relu(x2)
        x2 = self.fc10(x2)
        x2 = torch.sigmoid(x2)
        
        return x1, x2

