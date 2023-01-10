import os
p = os.getenv('PATH')
ld = os.getenv('LD_LIBRARY_PATH')
os.environ['PATH'] = f"/usr/local/cuda-11.1/bin:{p}"
os.environ['LD_LIBRARY_PATH'] = f"/usr/local/cuda-11.1/lib64:{ld}"
from google.colab.patches import cv2_imshow
from sklearn.metrics import jaccard_score
from PIL import Image, ImageDraw
from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
import datetime
import random
import json
import cv2
import csv
import os
import matplotlib.pyplot as plt


# import some common pytorch utilities
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch

from dataLoader import *

# import some common detectron2 utilities
import detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import GenericMask 
from detectron2.data import build_detection_test_loader
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from collections import defaultdict
setup_logger()

from detectron2.data import detection_utils as utils
from detectron2.data import build_detection_test_loader, build_detection_train_loader
import detectron2.data.transforms as T
import copy

# Define the location of current directory, which should contain data/train, data/test, and data/train.json.
# TODO: approx 1 line
#BASE_DIR = '/content/drive/My Drive/Colab Notebooks/03-cnn-detection-segmentation'
BASE_DIR = './'
train_dir = './data/train'
test_dir = './data/test'
#BASE_DIR = '/content/CMPT_CV_lab3'
OUTPUT_DIR = '{}/output'.format(BASE_DIR)
H5_DIR = '{}/h5'.format(BASE_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)


class PlaneDataset(Dataset):
    '''
    # We have provided a template data loader for your segmentation training
    # You need to complete the __getitem__() function before running the code
    # You may also need to add data augmentation or normalization in here
    '''
    def __init__(self, set_name, data_list):
        self.transforms = transforms.Compose([
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomVerticalFlip(),
            transforms.ToTensor() # Converting the image to tensor and change the image format (Channels-Last => Channels-First)
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomVerticalFlip()
        ])
        self.set_name = set_name
        self.data = data_list
        self.instance_map = []
        for i, d in enumerate(self.data):
            for j in range(len(d['annotations'])):
            self.instance_map.append([i,j])

    '''
    # you can change the value of length to a small number like 10 for debugging of your training procedure and overfeating
    # make sure to use the correct length for the final training
    '''
    def __len__(self):
        return len(self.instance_map)

    def numpy_to_tensor(self, img, mask):
        if self.transforms is not None:
            img = self.transforms(img)
        img = torch.tensor(img, dtype=torch.float)
        mask = torch.tensor(mask, dtype=torch.float)
        return img, mask

    '''
    # Complete this part by using get_instance_sample function
    # make sure to resize the img and mask to a fixed size (for example 128*128)
    # you can use "interpolate" function of pytorch or "numpy.resize"    
    '''
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        idx = self.instance_map[idx]
        data = self.data[idx[0]]
        
        img, mask = get_instance_sample(data, idx[1])    
        img = cv2.resize(img, (128, 128))
        mask = cv2.resize(mask, (128, 128))
        img, mask = self.numpy_to_tensor(img, mask)
        mask = torch.unsqueeze(mask,0)

        return img, mask

    def get_plane_dataset(set_name='train', batch_size=2):
        my_data_list = DatasetCatalog.get("data_detection_{}".format(set_name))
        dataset = PlaneDataset(set_name, my_data_list)
        
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=4,
                                                pin_memory=True, shuffle=True)
        return loader, dataset


'''
# convolution module as a template layer consists of conv2d layer, batch normalization, and relu activation
'''
class conv(nn.Module):
    def __init__(self, in_ch, out_ch, activation=True):
        super(conv, self).__init__()
        if(activation):
          self.layer = nn.Sequential(
             nn.Conv2d(in_ch, out_ch, 3, padding=1),
             nn.BatchNorm2d(out_ch),
             nn.ReLU(inplace=True)
          )
        else:
          self.layer = nn.Sequential(
             nn.Conv2d(in_ch, out_ch, 3, padding=1)  
             )

    def forward(self, x):
        x = self.layer(x)
        return x

'''
# convolution module as a template layer consists of conv2d layer, batch normalization, and relu activation
'''
class conv2(nn.Module):
    def __init__(self, in_ch, out_ch, activation=True):
        super(conv2, self).__init__()
        if(activation):
          self.layer = nn.Sequential(
             nn.Conv2d(in_ch, out_ch, 3, padding=1),
             nn.BatchNorm2d(out_ch),
             nn.ReLU(inplace=True),
             nn.Conv2d(in_ch, out_ch, 3, padding=1),
             nn.BatchNorm2d(out_ch),
             nn.ReLU(inplace=True)
          )
        else:
          self.layer = nn.Sequential(
             nn.Conv2d(in_ch, out_ch, 3, padding=1)  
             )

    def forward(self, x):
        x = self.layer(x)
        return x

'''
# downsampling module equal to a conv module followed by a max-pool layer
'''
class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.layer = nn.Sequential(
            #conv(in_ch, out_ch),
            nn.MaxPool2d(2)
            )

    def forward(self, x):
        x = self.layer(x)
        return x

'''
# upsampling module equal to a upsample function followed by a conv module
'''
class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)

        self.conv = conv(in_ch, out_ch)
        #self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        y = self.up(x)
        #y = self.dropout(y)
        #y = self.conv(y)
        return y

class Concatenate(nn.Module):
    def __init__(self):
        super(Concatenate, self).__init__()

    def forward(self, layer_1, layer_2):
        x = torch.cat((layer_1, layer_2), 1)

        return x
'''
# the main model which you need to complete by using above modules.
# you can also modify the above modules in order to improve your results.
'''
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
        # Encoder
        self.concat = Concatenate()
        self.concat2 = Concatenate()
        self.concat3 = Concatenate()
        self.concat4 = Concatenate()

        self.dropout = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        self.dropout4 = nn.Dropout(0.1)
        self.dropout5 = nn.Dropout(0.1)
        self.dropout6 = nn.Dropout(0.1)
        self.dropout7 = nn.Dropout(0.1)
        self.dropout8 = nn.Dropout(0.1)        
        
        self.input_conv = conv(3, 32)
        self.conv1 = conv(32, 32)
        self.down1 = down(32, 64)
        self.conv2 = conv(32, 64)
        self.conv2_a = conv(64, 64)
        self.down2 = down(64, 128)
        self.conv3 = conv(64, 128)
        self.conv3_a = conv(128, 128)
        self.down3 = down(128, 256)
        self.conv4 = conv(128, 256)
        self.conv4_a = conv(256, 256)
        self.down4 = down(256, 512)
        self.conv5 = conv(256, 512)
        self.conv5_a = conv(512, 512)
        
       
        # Decoder
        self.up1 = up(512, 256)
        self.conv6 = conv(256 + 512, 256)
        self.conv6_a = conv(256, 256)
        self.up2 = up(256, 128)
        self.conv7 = conv(128 + 256, 128)
        self.conv7_a = conv(128, 128)
        self.up3 = up(128, 64)
        self.conv8 = conv(64 + 128, 64)
        self.conv8_a = conv(64, 64)
        self.up4 = up(64, 32)
        self.conv9 = conv(32 + 64, 32)
        self.conv9_a = conv(32, 32)
        self.output_conv = conv(32, 1, False) # ReLu activation is removed to keep the logits for the loss function
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, input):
      y = self.input_conv(input)
      c1 = self.conv1(y)
      p1 = self.down1(c1)
      p1 = self.dropout(p1)
      
      c2 = self.conv2(p1)
      c2 = self.conv2_a(c2)
      p2 = self.down2(c2)
      p2 = self.dropout2(p2)

      c3 = self.conv3(p2)
      c3 = self.conv3_a(c3)
      p3 = self.down3(c3)
      p3 = self.dropout3(p3)

      c4 = self.conv4(p3)
      c4 = self.conv4_a(c4)
      p4 = self.down4(c4)
      p4 = self.dropout4(p4)

      
      c5 = self.conv5(p4)
      c5 = self.conv5_a(c5)

      u6 = self.up1(c5)
      #print(u6.shape)
      u6 = self.concat(u6, c4)
      #u6 = torch.cat([u6, c4], dim=1)
      #print(u6.shape)
      u6 = self.dropout5(u6)      
      c6 = self.conv6(u6)
      c6 = self.conv6_a(c6)
      

      u7 = self.up2(c6)
      u7 = self.concat2(u7, c3)
      u7 = self.dropout6(u7)
      c7 = self.conv7(u7)
      c7 = self.conv7_a(c7)

      u8 = self.up3(c7)
      u8 = self.concat3(u8, c2)
      u8 = self.dropout7(u8)
      c8 = self.conv8(u8)
      c8_a = self.conv8_a(c8)

      u9 = self.up4(c8)
      u9 = self.concat4(u9, c1)
      u9 = self.dropout8(u9)
      c9 = self.conv9(u9)
      c9 = self.conv9_a(c9)

      y = self.output_conv(c9)
      y = self.sigmoid(y)      
      output = y
      
      return output



if __name__=='__main__':
    # Set the hyperparameters
    num_epochs = 25
    batch_size = 16
    learning_rate = 0.001
    #learning_rate = 0.001
    weight_decay = 1e-5

    model = MyModel() # initialize the model
    model = model.cuda() # move the model to GPU
    loader, _ = get_plane_dataset('train', batch_size) # initialize data_loader
    #crit = nn.BCEWithLogitsLoss() # Define the loss function
    #crit = nn.CrossEntropyLoss()
    crit = nn.MSELoss()
    #optim = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay) # Initialize the optimizer as SGD
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    acc_increase = 20

    crit = nn.BCEWithLogitsLoss() # Define the loss function
    loader, _ = get_plane_dataset('train', 32) # initialize data_loader
    iou_values = []
    loss_values = []
    curr_iou = 0.65
    # start the training procedure
    start_epoch = 0
    num_epochs = 50
    #loss_values = []

    for epoch in range(start_epoch, num_epochs):
        total_loss = 0
        for (img, mask) in tqdm(loader):
            #print(type(img))
            #print(img.dtype)
            img = torch.tensor(img, device=torch.device('cuda'), requires_grad = True)
            mask = torch.tensor(mask, device=torch.device('cuda'), requires_grad = True)
            pred = model(img)
            #pred = (pred > 0.5).int()
            loss = crit(pred, mask)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.cpu().data

            
        loss_values.append(total_loss/len(loader))
        print("Epoch: {}, Loss: {}".format(epoch, total_loss/len(loader)))
        
        val_iou = calculate_val_iou(model)
        print("\n #Epochs: {}, Mean IoU: {}".format(epoch, val_iou))
        iou_values.append(val_iou)

        #torch.save(model.state_dict(), '{}/output/{}_segmentation_model.pth'.format(BASE_DIR, epoch))

        if curr_iou > val_iou:
            acc_increase -= 1
        
        curr_iou = val_iou
        if acc_increase < 0 or val_iou > 0.8:
            break

    '''
    # Saving the final model
    '''
    torch.save(model.state_dict(), '{}/output/final_segmentation_model.pth'.format(BASE_DIR))