import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root="./dataset",train=False,transform=torchvision.transforms.ToTensor(),
                                       download=False)   #../表示父级目录。 ./表示当前目录

dataloader = DataLoader(dataset,batch_size=64)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()       #super()用于继承父类的属性
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=(3,3), stride=1, padding=0)

    def forward(self,x):
        x = self.conv1(x)
        return x

tudui = Tudui()
# print(tudui)

writer = SummaryWriter("logs/conv2d")
step = 0

#查看dataloader中的每个数据
#遍历
for data in dataloader:
    imgs,targets = data
    output = tudui(imgs)
    # print(imgs.shape)
    # print(output.shape)
    # torch.Size([64, 3, 32, 32])
    writer.add_images("input",imgs,step)
    # torch.Size([64, 6, 30, 30])  ->[xxx,3,30,30]
    output = torch.reshape(output,(-1,3,30,30))
    writer.add_images("output",output,step)
    step = step + 1

    if step > 155:
        break

writer.close()