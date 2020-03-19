"""
Long sentence
No context information
Consistent memory
"""
import torch.nn as nn

rnn = nn.RNN(100, 10)
print(rnn._parameters.keys())

"""
torch.Size([10, 10]) torch.Size([10, 100])
torch.Size([10]) torch.Size([10])
"""
print(rnn.weight_hh_l0.shape,rnn.weight_ih_l0.shape,)
print(rnn.bias_hh_l0.shape,rnn.bias_ih_l0.shape,)

rnn = nn.RNN(100, 10, 2)
print(rnn._parameters.keys())

"""
torch.Size([10, 10]) torch.Size([10, 100])
torch.Size([10]) torch.Size([10])
"""
print(rnn.weight_hh_l0.shape,rnn.weight_ih_l0.shape,)
print(rnn.bias_hh_l0.shape,rnn.bias_ih_l0.shape,)