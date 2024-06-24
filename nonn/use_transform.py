from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

img_path="data/train/ants_image/0013035.jpg"
img=Image.open(img_path)

writer=SummaryWriter("logs")

#transforms该如何使用
tensor_trans=transforms.ToTensor()
tensor_img=tensor_trans(img)

#将图像数据写入 TensorBoard 日志文件中，以便在 TensorBoard 中进行可视化展示
writer.add_image("Tensor_img",tensor_img)
writer.close()
# print(tensor_img)