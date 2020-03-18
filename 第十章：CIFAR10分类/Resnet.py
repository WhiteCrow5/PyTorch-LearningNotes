import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlk(nn.Module):
    '''
    resnet block
    '''
    def __init__(self, ch_in, ch_out, stride=1):
        '''
        :param ch_in:
        :param ch_out:
        '''
        super(ResBlk, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.extra = nn.Sequential()
        if ch_out != ch_in:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=stride),
                nn.BatchNorm2d(ch_out),
            )

    def forward(self, x):
        """

        :param x: [b, ch, h, w]
        :return:
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # short cut
        # extra [b, ch_in, h, w] --> [b, ch_out, h, w]
        #element - wise
        out = self.extra(x) + out
        out = F.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 3, 1),
            nn.BatchNorm2d(64)
        )
        # followed 4 blocks
        # [b, 64, h, w] -> [b, 128, h, 2]
        self.blk1 = ResBlk(64, 128, stride=2)
        # [b, 128, h, w] -> [b, 256, h, 2]
        self.blk2 = ResBlk(128, 256, stride=2)
        # [b, 256, h, w] -> [b, 512, h, 2]
        self.blk3 = ResBlk(256, 512, stride=2)
        # [b, 512, h, w] -> [b, 1024, h, 2]
        self.blk4 = ResBlk(512, 512, stride=2)
        self.outlayer = nn.Linear(512*1*1, 10)

    def forward(self, x):
        '''

        :param x:
        :return:
        '''
        x = F.relu(self.conv1(x))
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)
        x = F.adaptive_avg_pool2d(x, [1, 1])
        x = x.view(x.size(0), -1)
        x = self.outlayer(x)
        return x

if __name__ == '__main__':
    blk = ResBlk(64, 128, stride=2)
    tmp = torch.randn(2, 64, 32, 32)
    out = blk(tmp)
    print(out.shape)

    x = torch.randn(2,3,32,32)
    model = ResNet18()
    out = model(x)
    print(out.shape)








