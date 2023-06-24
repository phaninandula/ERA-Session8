import torch.nn.functional as F
import torch.nn as nn
from torchsummary import summary
import numpy as np

dropout_rate = 0.01

def norm_func(norm_method, out_channels):
    print(f'Method of Normalization : {norm_method}')
    if norm_method =='BN':
        return nn.BatchNorm2d(out_channels)
    else:
        print(f'Method of Normalization : {norm_method}')
        #n_groups = int(out_channels*0.5) #--> not working
        group = 3 if norm_method == 'GN' else 1
        print(f'Number of groups : {group}')
        return nn.GroupNorm(group, out_channels) # GroupNorm(1,out_channels) == Layer Norm

class CIFAR10Net(nn.Module):
    
    def __init__(self, norm_method='BN'):
        super(CIFAR10Net, self).__init__()

        # Conv_Block1
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=4, kernel_size=(3,3), padding=1, bias=False),
            nn.ReLU(),
            norm_func(norm_method,4),
            nn.Dropout(dropout_rate)
        ) #Output_size = 32

        # Conv_Block2
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(3,3), padding=1, bias=False),
            nn.ReLU(),
            norm_func(norm_method,8),
            nn.Dropout(dropout_rate)
        ) #Output_size = 32

        # Conv_Block_3
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(1,1), padding=0, bias=False),
            nn.ReLU(),
            norm_func(norm_method,16),
            nn.Dropout(dropout_rate)
        ) #Output_size = 32

        # Max Pool Layer1
        self.maxpool1 = nn.Sequential(
            nn.MaxPool2d(2,2)
        ) #Output_size = 16

        # Conv_Block3
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=20, kernel_size=(3,3), padding=1, bias=False),
            nn.ReLU(),
            norm_func(norm_method,20),
            nn.Dropout(dropout_rate)
        )  #Output_size = 16

        # Conv_Block4
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=24, kernel_size=(3,3), padding=1, bias=False),
            nn.ReLU(),
            norm_func(norm_method,24),
            nn.Dropout(dropout_rate)
        )  #Output_size = 16

        # Conv_Block5
        self.conv_block5 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=28, kernel_size=(3,3), padding=0, bias=False),
            nn.ReLU(),
            norm_func(norm_method,28),
            nn.Dropout(dropout_rate)
        )  #Output_size = 14

        # Conv_Block_6
        self.conv_block_6 = nn.Sequential(
            nn.Conv2d(in_channels=28, out_channels=32, kernel_size=(1,1), padding=0, bias=False),
            nn.ReLU(),
            norm_func(norm_method,32),
            nn.Dropout(dropout_rate)
        )  #Output_size = 14

        # Max Pool Layer2
        self.maxpool2 = nn.Sequential(
            nn.MaxPool2d(2,2)
        )  #Output_size = 7

        # Conv_Block7
        self.conv_block7 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=36, kernel_size=(3,3), padding=1, bias=False),
            nn.ReLU(),
            norm_func(norm_method,36),
            nn.Dropout(dropout_rate)
        )  #Output_size = 7

        # Conv_Block8
        self.conv_block8 = nn.Sequential(
            nn.Conv2d(in_channels=36, out_channels=40, kernel_size=(3,3), padding=1, bias=False),
            nn.ReLU(),
            norm_func(norm_method,40),
            nn.Dropout(dropout_rate)
        )  #Output_size = 7

        # Conv_Block9
        self.conv_block9 = nn.Sequential(
            nn.Conv2d(in_channels=40, out_channels=16, kernel_size=(3,3), padding=0, bias=False),
            nn.ReLU(),
            norm_func(norm_method,16),
            nn.Dropout(dropout_rate)
        )  #Output_size = 5

        # GAP Block
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=4)
        )

        # Conv_Block10
        self.conv_block10 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1,1), padding=0, bias=False),
            nn.ReLU(),
            #norm_func(norm_method,4),
        #nn.Dropout(dropout_rate)  ---> Normalization & Dropout should not be applied in the last layer of the network
        )

    def forward(self, x):
        
        x = self.conv_block1(x)
        x = self.conv_block2(x)

        x = self.conv_block_3(x)
        x = self.maxpool1(x)

        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)

        x = self.conv_block_6(x)
        x = self.maxpool2(x)

        x = self.conv_block7(x)
        x = self.conv_block8(x)
        x = self.conv_block9(x)

        x = self.gap(x)

        x = self.conv_block10(x)

        x = x.view(-1,10)

        return F.log_softmax(x, dim=-1)