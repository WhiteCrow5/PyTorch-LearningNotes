import torch
import numpy as np
import random
from torch import nn,optim,autograd

h_dim = 400
batch_size = 512
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, 2)
        )
    def forward(self, z):
        output = self.net(z)
        return output

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        output = self.net(x)
        return output.view(-1)

def data_generator():
    scale = 22.
    centers = [
        (1,0),
        (-1,0),
        (0,1),
        (0,-1),
        (1. / np.sqrt(2), 1./np.sqrt(2)),
        (1. / np.sqrt(2), -1./np.sqrt(2)),
        (-1. / np.sqrt(2), 1./np.sqrt(2)),
        (-1. / np.sqrt(2), -1./np.sqrt(2))
    ]
    centers = [(scale * x, scale * y) for x, y in centers]

    while True:
        dataset = []
        for i in range(batch_size):
            point = np.random.randn(2) * 0.02
            center = random.choice(centers)
            # N(0, 1) + center x1/x2
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset).astype(np.float32)
        dataset /= 1.414
        yield dataset

if __name__ == '__main__':
    torch.manual_seed(23)
    np.random.seed(23)
    data_iter = data_generator()
    x = next(data_iter)
    # print(x.shape)
    G = Generator().cuda()
    D = Discriminator().cuda()
    # print(G)
    # print(D)
    optimizer_G = optim.Adam(G.parameters(),lr = 5e-4, betas=(0.5, 0.9))
    optimizer_D = optim.Adam(D.parameters(),lr = 5e-4, betas=(0.5, 0.9))

    for epoch in range(50000):
        # 1. train Discrimator
        for _ in range(5):
            xr = next(data_iter)
            xr = torch.from_numpy(xr).cuda()
            predr = D(xr)
            lossr = predr.mean()
        # 1.2. train on fake data
        z = torch.randn(batch_size,2).cuda()
        xf = G(z)
        predf = D(xf)
        lossf = predf.mean()

        loss_D = lossr + lossf
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # 2. train Generator
        z = torch.randn(batch_size, 2).cuda()
        xf = G(z)
        predf = D(xf)
        loss_G = -predf.mean()
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        if epoch % 100 == 0:
            print(loss_G.item(),loss_D.item())


