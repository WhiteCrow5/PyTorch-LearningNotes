import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.randn(1,1,28,28)
layer = nn.Conv2d(1,3,kernel_size=3, stride=1, padding=0)

out = layer.forward(x)
print(out.shape)

layer = nn.Conv2d(1,3,kernel_size=3, stride=1, padding=1)
out = layer.forward(x)
print(out.shape)

layer = nn.Conv2d(1,3,kernel_size=3, stride=2, padding=1)
out = layer.forward(x)
print(out.shape)

print(layer.weight)

# batch norm
x = torch.rand(100,16,784)
layer = nn.BatchNorm1d(16)
out = layer(x)
print(layer.running_mean)
print(layer.running_var)

x = torch.rand(1, 16, 7, 7)
layer = nn.BatchNorm2d(16)
print(layer(x).shape)

print(layer.weight)

class ResBlk(nn.Module):
    def __init__(self, ch_in, ch_out):
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.extra = nn.Sequential()
        if ch_out != ch_in:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1),
                nn.BatchNorm2d(ch_out)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.extra(x) + out
        return out