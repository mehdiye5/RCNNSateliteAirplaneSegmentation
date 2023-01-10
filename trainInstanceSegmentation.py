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

import torchvision.transforms as transforms


# import some common pytorch utilities
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch

from dataLoader import *
from semanticSegmentation import *

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


'''
# Define a new function to obtain the prediction mask by passing a sample data
# For this part, you need to use all the previous parts (predictor, get_instance_sample, data preprocessings, etc)
# It is better to keep everything (as well as the output of this funcion) on gpu as tensors to speed up the operations.
# pred_mask is the instance segmentation result and should have different values for different planes.
# TODO: approx 35 lines
'''
import torch.nn.functional as Fun
def get_prediction_mask(data):
  transform = transforms.Compose([
          transforms.ToTensor(), # Converting the image to tensor and change the image format (Channels-Last => Channels-First)
      ])
  

  model = MyModel().cuda()
  model.load_state_dict(torch.load('{}/output/final_segmentation_model.pth'.format(BASE_DIR)))
  model = model.eval() # chaning the model to evaluation mode will fix the bachnorm layers

  d_type = "train" if len(data["annotations"]) > 0 else "test"

  if d_type == "train":
    img = train_dict[data["file_name"]]
  else:
    img = test_dict[data["file_name"]]

  
  pred_mask = np.zeros((img.shape[0], img.shape[1]))
  gt_mask = np.zeros((img.shape[0], img.shape[1]))
  
  outputs = predictor(img)
  annos = outputs["instances"].to("cpu")
  
  bbox = annos.pred_boxes.tensor  
  bbox = [np.round(i).int().numpy() for i in bbox]
  
  
  
  
  if len(data["annotations"]) > 0:
    const = 255 // len(data["annotations"])
  else:
    const = 255
  for i in range(len(data["annotations"])):
    obj_img, obj_mask = get_instance_sample(data, i)
    train_anno = data["annotations"][i]
    train_bbox = train_anno["bbox"]

    x = int(train_bbox[0])
    y = int(train_bbox[1])
    w = int(train_bbox[2])
    h = int(train_bbox[3])

    a, obj_mask = cv2.threshold(obj_mask, 0, 255, cv2.THRESH_BINARY, cv2.THRESH_OTSU)

    obj_mask = np.where(obj_mask > 0,obj_mask + const*i, obj_mask)
    gt_mask[y:y + h,x:x + w] = obj_mask
  
  
  k = 0  
  for box in bbox:
    x1 = box[0]
    y1 = box[1]
    x2 = box[2]
    y2 = box[3]

    crop = img[y1:y2, x1:x2]
    shape = (crop.shape[0], crop.shape[1])
    
    crop = cv2.resize(crop, (128,128))

    crop = transform(crop)
    crop = torch.tensor(crop, dtype=torch.float).unsqueeze(0).cuda()
    
    pred = model(crop)
    
    pred = pred.squeeze(0).squeeze(0).detach().cpu()
    pred = pred > 0.5
    
    pred = pred.int().numpy()
    
    trans_size = transforms.Compose([
          transforms.ToPILImage(),
          transforms.Resize(size=shape),
          transforms.ToTensor() # Converting the image to tensor and change the image format (Channels-Last => Channels-First)
      ])
    
    pred = trans_size(pred).squeeze(0).numpy()
    

    pred = np.where(pred > 0,pred + k, pred)
    
    
    pred_mask[y1:y2, x1:x2] = pred

    k +=1
      
  pred_mask = torch.from_numpy(pred_mask).cuda()
  gt_mask = torch.from_numpy(gt_mask).cuda()
  
  return img, gt_mask, pred_mask # gt_mask could be all zero when the ground truth is not given.


def rle_encoding(x):
    '''
    x: pytorch tensor on gpu, 1 - mask, 0 - background
    Returns run length as list
    '''
    dots = torch.where(torch.flatten(x.long())==1)[0]
    if(len(dots)==0):
      return []
    inds = torch.where(dots[1:]!=dots[:-1]+1)[0]+1
    inds = torch.cat((torch.tensor([0], device=torch.device('cuda'), dtype=torch.long), inds))
    tmpdots = dots[inds]
    inds = torch.cat((inds, torch.tensor([len(dots)], device=torch.device('cuda'))))
    inds = inds[1:] - inds[:-1]
    runs = torch.cat((tmpdots, inds)).reshape((2,-1))
    runs = torch.flatten(torch.transpose(runs, 0, 1)).cpu().data.numpy()
    return ' '.join([str(i) for i in runs])

if __name__ == '__main__':
    im = [4,25,35]
    for i in range(3):
        img, gt_mask, pred_mask = get_prediction_mask(test_set[im[i]])
        plt.figure(figsize=(15,15))
        plt.subplot(i + 1, 2, 1)
        plt.imshow(img)  
        plt.subplot(i + 1, 2, 2)
        plt.imshow(pred_mask.detach().cpu().numpy())


    
    '''
    # You need to upload the csv file on kaggle
    # The speed of your code in the previous parts highly affects the running time of this part
    '''

    preddic = {"ImageId": [], "EncodedPixels": []}

    '''
    # Writing the predictions of the training set
    '''
    my_data_list = DatasetCatalog.get("data_detection_{}".format('train'))
    for i in tqdm(range(len(my_data_list)), position=0, leave=True):
        sample = my_data_list[i]
        sample['image_id'] = sample['file_name'].split("/")[-1][:-4]
        img, true_mask, pred_mask = get_prediction_mask(sample)
        inds = torch.unique(pred_mask)
        if(len(inds)==1):
            preddic['ImageId'].append(sample['image_id'])
            preddic['EncodedPixels'].append([])
        else:
            for index in inds:
                if(index == 0):
                    continue
                tmp_mask = (pred_mask==index)
                encPix = rle_encoding(tmp_mask)
                preddic['ImageId'].append(sample['image_id'])
                preddic['EncodedPixels'].append(encPix)

    '''
    # Writing the predictions of the test set
    '''

    my_data_list = DatasetCatalog.get("data_detection_{}".format('test'))
    for i in tqdm(range(len(my_data_list)), position=0, leave=True):
        sample = my_data_list[i]
        sample['image_id'] = sample['file_name'].split("/")[-1][:-4]
        img, true_mask, pred_mask = get_prediction_mask(sample)
        inds = torch.unique(pred_mask)
        if(len(inds)==1):
            preddic['ImageId'].append(sample['image_id'])
            preddic['EncodedPixels'].append([])
        else:
            for j, index in enumerate(inds):
                if(index == 0):
                    continue
                tmp_mask = (pred_mask==index).double()
                encPix = rle_encoding(tmp_mask)
                preddic['ImageId'].append(sample['image_id'])
                preddic['EncodedPixels'].append(encPix)

    pred_file = open("{}/pred.csv".format(BASE_DIR), 'w')
    pd.DataFrame(preddic).to_csv(pred_file, index=False)
    pred_file.close()