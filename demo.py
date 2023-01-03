# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 01:34:14 2022

@author: Frane
"""

import torch
import torchvision
from utils.io import Img_to_zero_center,Reverse_zero_center
from mtcnn.mtcnn import MTCNN
from PIL import Image
import numpy as np
import cv2
from skimage import transform as trans
from model.faceAlexnet import AgeAlexNet
from model.GAN import IPCGANs

class Demo:
    def __init__(self,generator_state_pth):
        #self.model = AgeAlexNet(True,generator_state_pth)
        self.model = IPCGANs()
        dict_state = torch.load(generator_state_pth)
        self.model.load_generator_state_dict(dict_state)
        


    def demo(self,image,target=0):
        assert target<5 and target>=0, "label shoule be less than 5"

        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((128,128)),
            torchvision.transforms.ToTensor(),
            Img_to_zero_center()
        ])
        label_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        image=transforms(image).unsqueeze(0)
        full_one = np.ones((128, 128), dtype=np.float32)
        full_zero = np.zeros((128, 128, 5), dtype=np.float32)
        full_zero[:, :, target] = full_one
        label=label_transforms(full_zero).unsqueeze(0)
        
        

        img=image.cuda()
        lbl=label.cuda()
        self.model.cuda()

        
        res=self.model.test_generate(img, lbl)
        res=Reverse_zero_center()(res)
        res_img=res.squeeze(0).cpu().numpy().transpose(1,2,0)
        return Image.fromarray((res_img*255).astype(np.uint8))
        #return(np.argmax(np.exp(res)/np.exp(res).sum()))

if __name__ == '__main__':
    #D=Demo("C:/Users/Frane/Desktop/DIPLOMSKI/peta/duboko_ucenje2/projekt/checkpoint/pretrain_alexnet/saved_parameters/epoch_20_iter_0.pth")
    D=Demo("C:/Users/Frane/Desktop/DIPLOMSKI/peta/duboko_ucenje2/projekt/checkpoint/IPCGANS/2022-12-26_18-10-32/saved_parameters/gepoch_2_iter_3000.pth")
    img=Image.open("C:/Users/Frane/Pictures/kiki2.jpg")
    img.show()
    res = D.demo(img, 0)
    res.show()
    """if(res == 0):
        print("predviđam 0-20 god")
    elif(res == 1):
        print("predviđam 20-29 god")
    elif(res == 2):
        print("predviđam 30-39 god")
    elif(res == 3):
        print("predviđam 40-49 god")
    else:
        print("predviđam više od 50 god")"""
    
#C:/Users/Frane/Desktop/DIPLOMSKI/peta/duboko_ucenje2/projekt/CACD2000/62_William_Katt_0007.jpg