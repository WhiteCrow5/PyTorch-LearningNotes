import torch.nn as nn
import torch

lstm = nn.LSTM(input_size=100,hidden_size=20,num_layers=4)
print(lstm)

x = torch.randn(10,3,100)
out,(h,c) = lstm(x)
print(out.shape, h.shape, out.shape)

print('one layer lstm')
cell = nn.LSTMCell(input_size=100, hidden_size=20)
h = torch.zeros(3, 20)
c = torch.zeros(3, 20)
for xt in x:
    h, c = cell(xt, [h, c])

# torch.Size([3, 20]) torch.Size([3, 20])
print(h.shape, c.shape)

print('two layer lstm')
cell1 = nn.LSTMCell(input_size=100, hidden_size=30)
cell2 = nn.LSTMCell(input_size=30, hidden_size=20)
h1 = torch.zeros(3, 30)
c1 = torch.zeros(3, 30)
h2 = torch.zeros(3, 20)
c2 = torch.zeros(3, 20)
for xt in x:
    h1, c1 = cell1(xt, [h1, c1])
    h2, c2 = cell2(h1, [h2, c2])

# torch.Size([3, 20]) torch.Size([3, 20])
print(h2.shape, c2.shape)