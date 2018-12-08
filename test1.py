'''
TO test the Yolo_detect class
'''
from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
from darknet import Darknet
from preprocess import inp_to_image
import pandas as pd
import random
import argparse
import pickle as pkl
from PIL import Image
from Yolo_detect import Detector

image=cv2.imread('imgs/messi.jpg')
# image=Image.open('imgs/messi.jpg')
# image=np.array(image)
d=Detector(0.5,0.3,416,image)
d.detector()
