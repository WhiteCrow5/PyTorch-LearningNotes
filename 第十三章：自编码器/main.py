import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from ae import AE
from vae import VAE
from torch import nn, optim

if __name__ == '__main__':
    mnist_train = datasets.MNIST('data', True, transform=transforms.Compose([
        transforms.ToTensor(),
    ]), download=True)
    mnist_train = DataLoader(mnist_train, batch_size=32, shuffle=True)
    mnist_test = datasets.MNIST('data', False, transform=transforms.Compose([
        transforms.ToTensor(),
    ]), download=True)
    mnist_test = DataLoader(mnist_test, batch_size=32, shuffle=True)

    x, _ = iter(mnist_train).next()
    print('x:',x.shape)

    device = torch.device('cuda')
    # model = AE().to(device)
    model = VAE().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1000):
        for batchidx, (x, _) in enumerate(mnist_train):
            x = x.to(device)
            x_hat, kld = model(x)
            loss = criterion(x_hat,x)

            if kld is not None:
                elbo = -loss - kld * 1.0
                loss = -elbo

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 50 == 0:
            print(epoch,'test loss:',loss.item(),kld.item())
            # x, _ = iter(mnist_test).next()
            # x = x.to(device)
            # with torch.no_grad():
            #     x_hat = model(x)
            #     loss = criterion(x_hat,x)
            #     print(epoch, 'train loss:', loss.item())

