import torch.nn as nn
import matplotlib.pyplot as plt
import torch

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)

def plot_image(img, label, name):
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(img[i][0]*0.308 + 0.1307, cmap='gray', interpolation=None)
        plt.title("{}:{}".format(name, label[i].item()))
        plt.xticks(), plt.yticks()
    plt.show()