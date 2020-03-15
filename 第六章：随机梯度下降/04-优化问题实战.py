import torch
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def himmelblau(x):
    return (x[0] ** 2 + x[1] -11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2
x = np.arange(-6, 6, 0.1)
y = np.arange(-6, 6, 0.1)
print('x,y range:',x.shape, y.shape)
X, Y = np.meshgrid(x, y)
print('X,Y maps:', x.shape, y.shape)
Z = himmelblau([X,Y])

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X,Y,Z)
ax.view_init(60,-30)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

#使用随机梯度下降方法求解
'''
f(3.2,2.0) = 0.0
f(-2.805118,3.131312) = 0.0
f(-3.779310,-3.283186) = 0.0
f(3.584428,-1.848126) = 0.0
'''
#[0,0]  --> 3.2,    2.0
#[4,0]  --> 3.584428310394287,  -1.8481265306472778
#[-4,0] --> -3.7793102264404297,    -3.2831859588623047
#[0,4]  --> -2.8051180839538574,    3.131312370300293
x = torch.tensor([0.,4.], requires_grad=True)
optimizer = torch.optim.Adam([x], lr = 0.001)
for step in range(20000):
    pred = himmelblau(x)
    optimizer.zero_grad()
    pred.backward()
    optimizer.step()

    if step % 2000 == 0:
        print('step{}:x={}, fx={}'
              .format(step,x.tolist(),pred.item()))