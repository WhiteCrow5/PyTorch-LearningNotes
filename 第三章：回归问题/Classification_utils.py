import torch
import matplotlib.pyplot as plt

def plot_curve(data):
    '''
    绘制传入的数组的图像
    :param data:数组
    :return: None
    '''
    fig = plt.figure()
    plt.plot(range(len(data)), data, color = 'blue')
    plt.legend(['value'], loc = 'upper right')
    plt.xlabel('step')
    plt.ylabel('value')
    plt.show()

def plot_image(img, label, name):
    '''
    给定图片，返回识别结果
    :param img: 图片
    :param label: 标签
    :param name:
    :return:
    '''
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(img[i][0] * 0.3081 + 0.1307,
                   cmap='gray',interpolation='none')
        plt.title('{}:{}'.format(name, label[i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()

def one_hot(label, depth=10):
    '''
    给定标签，返回其one_hot编码
    :param label:
    :param depth:
    :return: None
    '''
    out = torch.zeros(label.size(0), depth)
    idx = torch.LongTensor(label).view(-1, 1)
    out.scatter_(dim=1, index=idx, value=1)
    return out
