import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *
import time

# 准备数据集
train_data = torchvision.datasets.CIFAR10("./dataset", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)


# length 长度
train_data_size = len(train_data)
test_data_size = len(test_data)

print("训练数据集的长度为：{}".format(train_data_size))
print(f"测试数据集的长度为：{test_data_size}")

# 利用Dataloder 来加载数据集
train_dataloader = DataLoader(dataset=train_data, batch_size=64)
test_dataloader = DataLoader(dataset=test_data,batch_size=64)

# 创建网络模型
tudui = Tudui()
# 损失函数
loss_fn = nn.CrossEntropyLoss()
# 优化器
# 1e-2 = (10)*(-2) = 1/100
learning_rate = 1e-2
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
total_train_step = 0
total_test_step = 0
epoch = 10      #训练的轮数

#添加tensorboard
writer = SummaryWriter("logs_train")
start_time = time.time()

for i in range(epoch):
    print(f"-----第{i+1}轮训练开始-----")

    #训练步骤开始
    tudui.train()
    for data in train_dataloader:
        imgs, targets = data
        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)
        # 优化器优化模型
        optimizer.zero_grad()  #初始化
        loss.backward()
        optimizer.step()
        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print(end_time-start_time)
            print(f"训练次数{total_train_step},loss：{loss.item()}")
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    #测试步骤开始
    tudui.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():         #不累计梯度
        for data in test_dataloader:
            imgs, targets = data
            outputs = tudui(imgs)
            loss = loss_fn(outputs,targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()     #横向，对应位置相等的个数
            total_accuracy = total_accuracy + accuracy

    print(f"整体测试集上的loss：{total_test_loss}")
    print(f"整体测试集上的正确率：{total_accuracy/test_data_size}")
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step += 1

    torch.save(tudui, "tudui_{}.pth".format(i))
    #torch.save(tudui.state_dict(),"tudui_{}.pth".format(i))
    print("模型已保存")

writer.close()
