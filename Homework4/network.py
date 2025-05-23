'''
Author: J0hNnY1ee joh1eenny@gmail.com
Date: 2025-05-06 16:27:24
LastEditors: J0hNnY1ee joh1eenny@gmail.com
LastEditTime: 2025-05-06 21:58:56
FilePath: /HITSZ-CV/Homework4/network.py
Description: 

Copyright (c) 2025 by J0hNnY1ee joh1eenny@gmail.com, All Rights Reserved. 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x) 
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.fc1(x)
        x = F.relu(x) 
        x = self.fc2(x)
        x = F.relu(x) 
        x = self.fc3(x)
        return x
    
    
    
class ModifiedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.bn_conv1 = nn.BatchNorm2d(6) 
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.bn_conv2 = nn.BatchNorm2d(16) 
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.bn_fc1 = nn.BatchNorm1d(120)
        

        self.fc2 = nn.Linear(120, 84)
        self.bn_fc2 = nn.BatchNorm1d(84)
         

        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn_conv1(x) 
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn_conv2(x) 
        x = F.relu(x)
        x = self.pool2(x)

        x = x.view(-1, 16 * 5 * 5)

        x = self.fc1(x)
        x = self.bn_fc1(x) 
        x = F.relu(x)

        x = self.fc2(x)
        x = self.bn_fc2(x) 
        x = F.relu(x)

        x = self.fc3(x)
        return x