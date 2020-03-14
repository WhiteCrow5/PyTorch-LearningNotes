'''
MNIST手写数字识别数据：
    each number owns 7000 images
    train/test spliting:60K VS 10K
    28 * 28 = 784
    X = [1,2,3,4,...,784]
'''

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import torchvision
from matplotlib import pyplot as plt
from Classification_utils import plot_image, plot_curve, one_hot

batch_size = 512
# 1. 加载数据
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.FashionMNIST('mnist_data',train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,),(0.3081, ))
                               ])),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.FashionMNIST('mnist_data',train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,),(0.3081, ))
                               ])),
    batch_size=batch_size, shuffle=True)

# 查看部分数据
# x,y = next(iter(train_loader))
# print(x.shape,y.shape,x.min(),x.max())
# plot_image(x,y,'img_sample')

# 2.定义网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # xw + b
        self.fc1 = nn.Linear(28*28,256)
        self.fc2 = nn.Linear(256,64)
        self.fc3 = nn.Linear(64,10)

    def forward(self, x):
        # x ; [b, 1, 28, 28]
        # h1 = relu(xw + b)
        x = F.relu(self.fc1(x))
        # h2 = relu(h1w + b)
        x = F.relu(self.fc2(x))
        # h3 = h2w3 + b3
        x = self.fc3(x)

        return x

# 实例化网络、优化器
net = Net()
# optimizer = optim.SGD(net.parameters(), lr = 0.005, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr = 0.005)

# 3. 训练模型
train_loss = []
for epoch in range(3):
    for batch_idx, (x,y) in enumerate(train_loader):
        # print(x.shape,y.shape)
        # torch.Size([512, 1, 28, 28]) torch.Size([512])

        x = x.view(x.size(0),-1)
        # print(x.shape,y.shape)
        # torch.Size([512, 784]) torch.Size([512])

        out = net(x)
        y_onehot = one_hot(y)

        # loss = mes(out, y_onehot)
        # loss = F.mse_loss(out,y_onehot)
        loss = F.cross_entropy(out,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        if batch_idx % 10 == 0:
            print('epoch:{}, batch_idx:{}, loss:{}'.format(epoch, batch_idx, loss))

# 打印损失函数
plot_curve(train_loss)

# 4. 计算模型在测试集上的效果
total_correct = 0
for x,y in test_loader:
    x = x.view(x.size(0), -1)
    out = net(x)
    # 预测的数字
    pred = out.argmax(dim=1)
    correct = pred.eq(y).sum().float().item()
    total_correct += correct

total_num = len(test_loader.dataset)
acc = total_correct / total_num
print('test_acc: ',acc)

x, y = next(iter(test_loader))
out = net(x.view(x.size(0),-1))
pred = out.argmax(dim=1)
plot_image(x,pred,'test')
