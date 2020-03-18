import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
# from lenet5 import LeNet5
from Resnet import ResNet18
from torch import nn as nn

def main():
    batch_size = 32
    cifar_train = datasets.CIFAR10('cifar',True,transform=transforms.Compose([
        transforms.Resize(32,32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225]),
    ]),download=True)
    cifar_train = DataLoader(cifar_train, batch_size=batch_size, shuffle=True)

    cifar_test = datasets.CIFAR10('cifar',False,transform=transforms.Compose([
        transforms.Resize(32,32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225]),
    ]),download=True)
    cifar_test = DataLoader(cifar_test, batch_size=batch_size, shuffle=True)

    # x, label = iter(cifar_train).__next__()
    # print('x:',x.shape, 'label:',label.shape)

    device = torch.device('cuda')
    # model = LeNet5().to(device)
    model = ResNet18().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(model)

    for epoch in range(5):
        for batch_idx,(x,label) in enumerate(cifar_train):
            #[b, 3, 32, 32]
            x,label = x.to(device), label.to(device)
            out = model(x)
            loss = criterion(out,label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('loss:',loss.item())

        # test
        model.eval()
        with torch.no_grad():
            total_correct = 0
            total_num = 0
            for x, label in cifar_test:
                x, label = x.to(device), label.to(device)
                out = model(x)
                pred = out.argmax(dim=1)
                total_correct += torch.eq(pred,label).float().sum()
                total_num += x.size(0)
            acc = total_correct/total_num
            print('acc:',acc)

if __name__ == '__main__':
    main()