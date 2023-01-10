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
# This function should return a list of data samples in which each sample is a dictionary. 
# Make sure to select the correct bbox_mode for the data
# For the test data, you only have access to the images, therefore, the annotations should be empty.
# Other values could be obtained from the image files.
# TODO: approx 35 lines
'''
def get_detection_data(set_name):
  data_dirs = '{}/data'.format(BASE_DIR)
  train_dir = '{}/data/train'.format(BASE_DIR)
  test_dir = '{}/data/test'.format(BASE_DIR)
  dataset = []
  
  if set_name == "test":
    
    for fname in os.listdir(test_dir):
      record = {}
      filename = os.path.join(test_dir, fname)

      #height, width = cv2.imread(filename).shape[:2]
      height, width = train_dict(filename).shape[:2]

      image_id = fname.split("P")[1]
      image_id = image_id.split(".png")[0]
      #print(image_id)
      image_id = int(image_id)

      record["file_name"] = filename
      record["image_id"] = image_id
      record["height"] = height
      record["width"] = width

      record["annotations"] = []
      dataset.append(record)
      
  else:
    json_file = os.path.join(data_dirs, "train.json")

    with open(json_file) as f:
          json_list = json.load(f)
    
    all_annotations = defaultdict(list)
    for i in json_list:
      json_fname = i["file_name"]
      all_annotations[json_fname].append(i)

    #for v in json_list:
    for idx, fname in enumerate(os.listdir(train_dir)):
      record = {}

      filename = os.path.join(train_dir, fname)
      #height, width = cv2.imread(filename).shape[:2]
      height, width = train_dict(filename).shape[:2]

      record["file_name"] = filename
      record["image_id"] = idx
      record["height"] = height
      record["width"] = width

      
      objs = []

      for anno in all_annotations[fname]:

        obj = {
              "bbox": anno["bbox"],
              "bbox_mode": BoxMode.XYWH_ABS,
              "segmentation": anno["segmentation"],
              "category_id": 0,
              "category_name": anno["category_name"],
              "iscrowd": anno["iscrowd"],
              "area": anno["area"],
          }
      
        objs.append(obj)
      
      record["annotations"] = objs
      dataset.append(record)
  
  return dataset


def getDataSet():
    """
    This method returns the training, validation and test set for training.
    """
    # split the data
    all_data = get_detection_data("train")
    split_size  = int(np.floor(len(all_data)*0.8))
    train_set = all_data[:split_size]
    val_set = all_data[split_size:]
    test_set = get_detection_data("test")

    return (train_set, val_set, test_set)

def registerDataCatalog(train_set, val_set, test_set):
    '''
    # Remember to add your dataset to DatasetCatalog and MetadataCatalog
    # Consdier "data_detection_train" and "data_detection_test" for registration
    # You can also add an optional "data_detection_val" for your validation by spliting the training data
    # TODO: approx 5 lines
    '''
    DatasetCatalog.remove("data_detection_val")
    MetadataCatalog.remove("data_detection_val")

    DatasetCatalog.remove("data_detection_train")
    MetadataCatalog.remove("data_detection_train")

    DatasetCatalog.remove("data_detection_test")
    MetadataCatalog.remove("data_detection_test")

    #classes = ['plane', 'plane', 'plane', 'plane', 'plane', 'plane', 'plane', 'plane', 'plane', 'plane', 'plane', 'plane', 'plane', 'plane', 'plane']

    #train_set = all_data

    DatasetCatalog.register("data_detection_val", lambda d="data_detection_val": val_set)
    MetadataCatalog.get("data_detection_val").set(thing_classes=['plane'])


    DatasetCatalog.register("data_detection_train", lambda d="data_detection_train": train_set)
    MetadataCatalog.get("data_detection_train").set(thing_classes=['plane'])

    DatasetCatalog.register("data_detection_test", lambda d="data_detection_test": test_set)
    MetadataCatalog.get("data_detection_test").set(thing_classes=['plane'])

    data_detection_metadata = MetadataCatalog.get("data_detection_train")


def custom_mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    #image = utils.read_image(dataset_dict["file_name"], format="BGR")
    image = train_dict[dataset_dict["file_name"]]
    transform_list = [
        T.Resize((800,600)),        
        T.RandomBrightness(0.8, 1.8),
        T.RandomContrast(0.6, 1.3),
        T.RandomSaturation(0.8, 1.4),
        T.RandomRotation(angle=[90, 90]),
        T.RandomLighting(0.7),
        T.RandomFlip(prob=0.4, horizontal=False, vertical=True),
    ]
    image, transforms = T.apply_transform_gens(transform_list, image)
    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]
    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)
    return dataset_dict


def get_instance_sample(data, idx, img=None):
    '''
    # Write a function that returns the cropped image and corresponding mask regarding the target bounding box
    # idx is the index of the target bbox in the data
    # high-resolution image could be passed or could be load from data['file_name']
    # You can use the mask attribute of detectron2.utils.visualizer.GenericMask 
    #     to convert the segmentation annotations to binary masks
    # TODO: approx 10 lines
    '''
    train_dir = '{}/data/train'.format(BASE_DIR)

    #img = cv2.imread(data["file_name"])
    img = train_dict[data["file_name"]]
    height, width = img.shape[:2]

    anno = data["annotations"][idx]
    bbox = anno["bbox"]  
    
    
    x = int(bbox[0])
    y = int(bbox[1])
    w = int(bbox[2])
    h = int(bbox[3])
    
    obj_img = img[y:y + h,x:x + w]  

    #obj_mask = GenericMask(img[:,:,0], height, width)  
    #obj_mask = obj_mask.polygons_to_mask(anno["segmentation"])
    #obj_mask = obj_mask[y:y + h,x:x + w]

    obj_mask = GenericMask(anno["segmentation"], height, width).mask
    obj_mask = obj_mask[y:y + h,x:x + w]  
    
    
    return obj_img, obj_mask