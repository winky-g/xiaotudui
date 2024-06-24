from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer=SummaryWriter("logs")
img_path="data/train/ants_image/0013035.jpg"
img_PIL=Image.open(img_path)
img_array=np.array(img_PIL)
# print(img_array.shape)

writer.add_image('testdata',img_array,1,dataformats='HWC')

# y=2x
for i in range(100):
    writer.add_scalar("y=2x",2*i,i)

writer.close()
# conda install tensorboard -i https://pypi.tuna.tsinghua.edu.cn/simple


# import torch
# print(torch.cuda.is_available())