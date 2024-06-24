import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()     #图片转换成totensor类型
])

train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=dataset_transform, download=False)
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=dataset_transform, download=False)
# print(test_set[0])        #查看第一张图片张量
# print(test_set.classes)     #种类

# img,target = test_set[0]
# print(img)
# print(target)
# print(test_set.classes[target])
# img.show()

writer = SummaryWriter("logs/p10")
for i in range(10):
    img,target = test_set[i]
    writer.add_image("test_set", img, i)

writer.close()