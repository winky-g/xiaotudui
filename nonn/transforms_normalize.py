from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer=SummaryWriter("logs")
img=Image.open("images/pytorch.png")
# print(img)

# Totensor   把numpy的图片形式转换为tensor（张量）的形式
trans_Totensor = transforms.ToTensor()
img_tensor = trans_Totensor(img)
writer.add_image("ToTensor",img_tensor)

# Normalize  以张量的形式
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([1,3,5],[3,5,1])    #均值，标准差
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Normalize",img_norm,1)

writer.close()