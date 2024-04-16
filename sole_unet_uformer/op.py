import argparse
import json
import os
from cv2 import merge
import cv2
import skimage
from skimage import morphology
import torch
import numpy as np


_EPS=1e-7

def get_args_from_json(json_file_path):
    args_dict={}    
    summary_filename = json_file_path
    with open(summary_filename) as f:
        summary_dict = json.load(fp=f)
    for key in summary_dict.keys():
        args_dict[key] = summary_dict[key]
    return args_dict
    
def save_args_to_json(args_dict,json_folder_path='config/',file_name='config1.json'):
    json_file_path=json_folder_path+file_name
    args_dict = json.dumps(args_dict, indent=2, separators=(',', ':'))
    with open(json_file_path,"w") as f:
        #json.dump(args_dict,f)
        f.write(args_dict)

def mkdir(path):
	folder = os.path.exists(path)
	if not folder:
		os.makedirs(path)

def adjust_gamma(image: torch.Tensor, gamma):
    #image is in shape of [B,C,H,W] and gamma is in shape [B]
    gamma=gamma.float()
    gamma_tensor=torch.ones_like(image)
    gamma_tensor=gamma.view(-1,1,1,1)*gamma_tensor
    image=torch.pow(image,gamma_tensor)
    out= torch.clamp(image, 0.0, 1.0)
    return out

def adjust_gamma_reverse(image: torch.Tensor, gamma):
    #gamma=torch.Tensor([gamma])
    gamma=1/gamma.float()
    gamma_tensor=torch.ones_like(image)
    gamma_tensor=gamma.view(-1,1,1,1)*gamma_tensor
    image=torch.pow(image,gamma_tensor)
    out= torch.clamp(image, 0.0, 1.0)
    return out

def predict_flare_from_6_channel(input_tensor,gamma):
    #the input is a tensor in [B,C,H,W], the C here is 6

    deflare_img=input_tensor[:,:3,:,:]
    flare_img_predicted=input_tensor[:,3:,:,:]

    merge_img_predicted_linear=adjust_gamma(deflare_img,gamma)+adjust_gamma(flare_img_predicted,gamma)
    merge_img_predicted=adjust_gamma_reverse(torch.clamp(merge_img_predicted_linear, 1e-7, 1.0),gamma)
    return deflare_img,flare_img_predicted,merge_img_predicted
