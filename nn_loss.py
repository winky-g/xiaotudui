import torch
from torch import nn
from torch.nn import L1Loss

inputs = torch.tensor([1,2,3],dtype=torch.float32)
targets = torch.tensor([1,2,5],dtype=torch.float32)

inputs = torch.reshape(inputs,[1,1,1,3])           #[N,C,H,W]
targets = torch.reshape(targets,[1,1,1,3])

loss = L1Loss(reduction='mean')
result = loss(inputs,targets)

loss_mse = nn.MSELoss()     #均方差
result_mse = loss_mse(inputs,targets)

print(result)
print(result_mse)

x = torch.tensor([0.,0.2,0.3])
y = torch.tensor([1])
x = torch.reshape(x,(1,3))          #(N,C) N:batch-size C:class
loss_cross = nn.CrossEntropyLoss()    #交叉熵
result_cross = loss_cross(x,y)
print(result_cross)