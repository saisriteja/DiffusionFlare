from heapq import merge
from uformer_unet import Uformer

import torch

def load_params(model_path):
    #  full_model=torch.load(model_path)
     full_model=torch.load(model_path, map_location=torch.device('cpu'))
     if 'params_ema' in full_model:
          return full_model['params_ema']
     elif 'params' in full_model:
          return full_model['params']
     else:
          return full_model


pretrain_dir = '/data/tmp_teja/prince_uformer/DiffusionFlare/net_g_last.pth'
model=Uformer(img_size=512,img_ch=3,output_ch=6)
model.load_state_dict(load_params(pretrain_dir))


import sys
sys.path.append('../')

from distutils import debug
import torch
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt


import os
debug_path = "debug"
os.makedirs(debug_path, exist_ok=True)


# Open the image using PIL
img = Image.open("test_psnr.jpeg")
img =img.resize((512, 512))
# Convert the PIL image to a NumPy array
img_array = np.array(img)
img_array=np.transpose(img_array, (2, 0, 1))

# Add a batch dimension
img_array_with_batch = np.expand_dims(img_array, axis=0)

# Convert the NumPy array to a PyTorch tensor
img_tensor = torch.from_numpy(img_array_with_batch)
img_tensor = img_tensor.float()
# Move the tensor to GPU if available
# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
img_tensor = img_tensor.to(device)

# Open the image using PIL
depth = Image.open("fdepth_colored.png")
depth =depth.resize((512, 512))
depth = depth.convert("RGB")
# Convert the PIL image to a NumPy array
depth_array = np.array(depth)
depth_array=np.transpose(depth_array, (2, 0, 1))

# Add a batch dimension
depth_array_with_batch = np.expand_dims(depth_array, axis=0)

# Convert the NumPy array to a PyTorch tensor
depth_tensor = torch.from_numpy(depth_array_with_batch)
depth_tensor = depth_tensor.float()
# Move the tensor to GPU if available
# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
img_tensor = img_tensor.to(device)
depth_tensor = depth_tensor.to(device)


# img_tensor = depth_tensor
img_tensor = img_tensor/255.0
depth_tensor = depth_tensor/255.0


# Forward pass
with torch.no_grad():
#     op, info = gen(img_tensor,depth_tensor)
     op, info = model(img_tensor)



import cv2
from PIL import Image, ImageChops

flare_path = 'debug/flare.png'
blend_path = 'debug/blend.png'

img_path = "test_psnr.jpeg"
merge_img = Image.open(img_path)


from op import predict_flare_from_6_channel

gamma=torch.Tensor([2.2])
deflare_img,flare_img_predicted,merge_img_predicted=predict_flare_from_6_channel(input_tensor=op, gamma=gamma)
# merge_img_predicted=merge_img_predicted.cpu().numpy()

# save the deflare image
deflare_img=deflare_img.cpu().numpy()
deflare_img=np.transpose(deflare_img[0], (1, 2, 0))
deflare_img=deflare_img*255
cv2.imwrite(flare_path, deflare_img)

# save the flare image
flare_img_predicted=flare_img_predicted.cpu().numpy()
flare_img_predicted=np.transpose(flare_img_predicted[0], (1, 2, 0))
flare_img_predicted=flare_img_predicted*255
cv2.imwrite(blend_path, flare_img_predicted)

for layer, f_maps in info.items():
     print(f"{layer} -> {f_maps.shape}")

     B, N, C = f_maps.shape
     # reshape it B, sqrt(N), sqrt(N), C
     f_maps = f_maps.view(B, int(N**0.5), int(N**0.5), C)
     # stack of all them in a grid of size np.sqrt(C)
     rows = np.sqrt(C)
     cols = np.sqrt(C)

     rows = int(rows)
     cols = int(cols)

     plt.figure(figsize=(rows, cols))
     for i in range(rows):
          for j in range(cols):
               plt.subplot(rows, cols, i*cols+j+1)
               plt.imshow(f_maps[0, :, :, i*cols+j].cpu().numpy(), cmap='gray')
               plt.axis('off')
     
     # save the fig
     plt.savefig(f"{debug_path}/{layer}.png")
     plt.close()
     # save the output of the neural network