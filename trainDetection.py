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

class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=custom_mapper)

def visualizePlanes(train_set):
    '''
    # Visualize some samples using Visualizer to make sure that the function works correctly    
    '''

    for d in random.sample(train_set, 3):
        #img = cv2.imread(d["file_name"])
        img = train_dict(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=data_detection_metadata, scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        cv2_imshow(out.get_image()[:, :, ::-1])

def visulizeRandomSamples():
     # Visualize the output for 3 random test samples

    data_detection_metadata_val = MetadataCatalog.get("data_detection_test")
    for d in random.sample(val_set, 3):    
        #im = cv2.imread(d["file_name"])
        im = train_dict(d["file_name"])
        outputs = predictor(im)
        #print(outputs)
        v = Visualizer(im[:, :, ::-1],
                    metadata=data_detection_metadata_val, 
                    scale=0.5 
                    #instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2_imshow(out.get_image()[:, :, ::-1])

if __name__ == '__main__':
    (train_set, val_set, test_set) = getDataSet()

    registerDataCatalog(train_set, val_set, test_set)
    
    # Set the configs for the detection part in here.    
    cfg = get_cfg()
    cfg.OUTPUT_DIR = "{}/output/".format(BASE_DIR)

    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("data_detection_train",)
    cfg.DATASETS.TEST = ()

    cfg.DATALOADER.NUM_WORKERS = 2

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")

    cfg.SOLVER.IMS_PER_BATCH = 2

    cfg.SOLVER.BASE_LR = 0.002  

    #cfg.SOLVER.MAX_ITER = 500
    cfg.SOLVER.MAX_ITER = 6000

    #cfg.SOLVER.STEPS = []        

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 

    # Create a CustomTrainer using the above config and train the model
    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # After training the model, you need to update cfg.MODEL.WEIGHTS

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    visulizeRandomSamples()

    # Use COCOEvaluator and build_detection_train_loader
    # You can save the output predictions using inference_on_dataset
    evaluator = COCOEvaluator("data_detection_val", output_dir=OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "data_detection_val")
    inference_on_dataset(predictor.model, val_loader, evaluator)