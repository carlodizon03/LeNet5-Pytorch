import torch
import torch.nn as nn
from torchsummary import summary
from ptflops import get_model_complexity_info
import numpy as np
import time 

class LeNet(nn.Module):
    def __init__(self,
                 in_channels =1 , 
                 out_channels = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 =  nn.Sequential( 
                                        nn.Conv2d(in_channels = self.in_channels, out_channels = 6, kernel_size = 5, padding = 2),
                                        nn.ReLU(),
                                        nn.MaxPool2d(2,2),
                                        nn.Dropout(0.01)

                                    )
        self.conv2 = nn.Sequential(
                                        nn.Conv2d(6, 16, kernel_size = 5),
                                        nn.ReLU(),
                                        nn.MaxPool2d(2,2),
                                        nn.Dropout(0.01)
                                    )

        self.linear1 = nn.Sequential(
                                        nn.Linear(16*5*5, 120),
                                        nn.ReLU(),
                                        nn.Dropout(0.01)
                                    )
        self.linear2 = nn.Sequential(
                                        nn.Linear(120, 84),
                                        nn.ReLU(),
                                        nn.Dropout(0.01)
                                    )
        self.linear3 = nn.Sequential(
                                        nn.Linear(84, self.out_channels),
                                        nn.Dropout(0.02),
                                        nn.LogSoftmax(dim = -1)
                                    )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, np.prod(x.size()[1:]))
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)

        return x


"""Load Cuda """
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
""""""""""""""""""
model = LeNet(1,6)
model.to(device)
summary(model,(1,28,28))
macs, params = get_model_complexity_info(model, (1,28,28), as_strings=False,
                                           print_per_layer_stat=False, verbose=False)
print('{:<30}  {:<15} '.format('Computational complexity: ', macs/1000000))

times = []
model.eval()

for i in range(10):
    start_time =time.time() 
    image = torch.rand([1,1,28,28]).to(device)
    pred = model(image)
    tlapse = time.time() - start_time
    print("Inference Elapsed Time:{0:0.4f}".format(tlapse))
    times.append(tlapse)
print("Mean Inference Time per image: {0:0.4f} s".format(np.mean(times)))

