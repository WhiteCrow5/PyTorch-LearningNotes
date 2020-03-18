import torch
from torch import nn
from torch.nn import functional as F

class LeNet5(nn.Module):
    '''
    for cifar10 dataset
    '''
    def __init__(self):
        super(LeNet5,self).__init__()
        self.conv = nn.Sequential(
            # x:[b, 3, 32, 32]
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2,stride=2,padding=0),
            #
            nn.Conv2d(6,16,kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            #
        )
        #flatten
        self.fc = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        '''
        :param self:
        :param x: [b, 3, 32, 32]
        :return:
        '''
        # [b, 3, 32, 32]  --> [b, 16, 5, 5]
        x = self.conv(x)
        # [b, 16, 5, 5]   --> [b,16*5*5]
        x = x.view(x.size(0),-1)
        # [b, 16*5*5]   --> [b, 10]
        x = self.fc(x)
        return x

if __name__ == '__main__':
    net = LeNet5()
    tmp = torch.randn(2,3,32,32)
    out = net(tmp)
    print('conv out：', out.shape)
