import torch
import torch.nn as nn
from torchsummary import summary
from ptflops import get_model_complexity_info
import numpy as np
import time 


class LeNet(nn.Module):
    def __init__(self,
                 in_channels =1 , 
                 out_channels = 10):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 =  nn.Sequential( 
                                        nn.Conv2d(self.in_channels, 6, 5),
                                        nn.ReLU(),
                                        nn.MaxPool2d(2),

                                    )
        self.conv2 = nn.Sequential(
                                        nn.Conv2d(6, 16, 5),
                                        nn.ReLU(),
                                        nn.MaxPool2d(2,2),
                                    )

        self.linear1 = nn.Sequential(
                                        nn.Linear(16*4*4, 120),
                                        nn.ReLU(),
                                    )
        self.linear2 = nn.Sequential(
                                        nn.Linear(120, 84),
                                        nn.ReLU(),
                                    )
        self.linear3 = nn.Sequential(
                                        nn.Linear(84, self.out_channels),

                                    )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, np.prod(x.size()[1:]))
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)

        return x
