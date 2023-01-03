# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 16:52:16 2022

@author: Frane
"""

import os
import random
image_dir="C:/Users/Frane/Desktop/DIPLOMSKI/peta/duboko_ucenje2/projekt/CACD2000"

#make train_label_pair file
root = 'C:/Users/Frane/Desktop/DIPLOMSKI/peta/duboko_ucenje2/projekt/'

with open(root + 'data/cacd2000-lists/train.txt','r') as f:
    lines = f.readlines()
    
lines = [s.strip() for s in lines]

options = [0,1,2,3,4]

train_labels = []
for line in lines:
    img, label = line.split()
    row = []
    row.append(label)
    rand_label = random.randint(0,4)
    while(rand_label == int(label)):
        rand_label = random.randint(0,4)
    
    row.append(str(rand_label))
    
    train_labels.append(row[0] + " " + row[1] + "\n")
    

with open(root + 'data/cacd2000-lists/train_label_pair.txt','w') as f:
    lines = f.writelines(train_labels) 