import torch
import torchvision
import torch.nn.functional as F

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data',train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,),(0.3081, ))
                               ])),
    batch_size=200, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data',train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,),(0.3081, ))
                               ])),
    batch_size=200, shuffle=True)

w1 = torch.randn(200,784,requires_grad=True)
b1 = torch.randn(200,requires_grad=True)
w2 = torch.randn(200,200,requires_grad=True)
b2 = torch.randn(200,requires_grad=True)
w3 = torch.randn(10,200,requires_grad=True)
b3 = torch.randn(10,requires_grad=True)

def forward(x):
    x = x@w1.t() + b1
    x = F.relu(x)
    x = x@w2.t() + b2
    x = F.relu(x)
    x = x@w3.t() + b3
    x = F.relu(x)
    return x

optimizer = torch.optim.SGD([w1,b1,w2,b2,w3,b3], lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(10):
    for batch_idx,(data, target) in enumerate(train_loader):
        data = data.view(-1, 28*28)
        logits = forward(data)
        loss = criterion(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

torchvision.models.DenseNet