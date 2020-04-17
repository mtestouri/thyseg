import os
import sys
import logging
import json
import pandas as pd
import numpy as np
from cytomine import Cytomine
from cytomine.models import ImageInstance, Annotation
import torch
from torch.utils.data import Dataset
from glob import glob
import random
import cv2

def download_dataset(filename, folder, imhw=512):
    # hide logging
    logger = logging.getLogger()
    logger.disabled = True
    print("downloading dataset..") 
    cred = json.load(open('credentials.json'))
    with Cytomine(host=cred['host'], 
                  public_key=cred['public_key'], 
                  private_key=cred['private_key']) as conn:
        # create folder
        if not os.path.exists(folder):
            os.makedirs(folder)
        # dataset descriptor
        array = pd.read_csv(filename, sep=';').to_numpy()
        # download the images
        i = 1
        for row in array:
            annotation = Annotation()
            annotation.id = int(row[0])
            annotation.fetch()
            image = ImageInstance()
            image.id = int(row[5])
            image.fetch()
            # convert the coordinates
            x = round(float(row[3])-(imhw/2))
            y = image.height - round(float(row[4])+(imhw/2))
            # download slice and corresponding mask 
            slice_image = image.reference_slice()
            slice_image.window(x, y, imhw, imhw, dest_pattern=folder
                               + "/" + str(i) + "_x.jpg")
            slice_image.window(x, y, imhw, imhw, dest_pattern=folder 
                               + "/" + str(i) + "_y.jpg",
                               mask=True,
                               terms=annotation.term)
            if i > 1:
                sys.stdout.write("\033[F")
            sys.stdout.write("\033[K")
            print(f'progress: {round(i/len(array)*100, 1)}%')
            i += 1
    print("dataset downloaded")

from shutil import copyfile

def split_dataset(folder, split=0.8, seed=42):
    #TODO validation set
    print("splitting dataset..")
    while folder[-1] == '/':
        folder = folder[:len(folder)-1]
    # load list of files
    files = glob(folder + "/*_x.jpg")
    if len(files) == 0:
        raise FileNotFoundError("no files found in folder '" + folder + "'")
    # shuffle the files
    random.Random(seed).shuffle(files)
    train_files = files[:round(split*len(files))]
    test_files = files[round(split*len(files)):]
    # create train folder
    train_f = folder + "_train"
    if not os.path.exists(train_f):
        os.makedirs(train_f)
    for x_file in train_files:
        y_file = folder + "/" + x_file[len(folder)+1:len(x_file)-6] + "_y.jpg"
        copyfile(x_file, train_f + "/" + x_file[len(folder)+1:])
        copyfile(y_file, train_f + "/" + y_file[len(folder)+1:])
    # create test folder
    test_f = folder + "_test"
    if not os.path.exists(test_f):
        os.makedirs(test_f)
    for x_file in test_files:
        y_file = folder + "/" + x_file[len(folder)+1:len(x_file)-6] + "_y.jpg"
        copyfile(x_file, test_f + "/" + x_file[len(folder)+1:])
        copyfile(y_file, test_f + "/" + y_file[len(folder)+1:])
    print("dataset split")

class ImgSet(Dataset):
    def __init__(self, folder):
        while folder[-1] == '/':
            folder = folder[:len(folder)-1]
        self.df = folder
        # load list of files
        self.files = glob(self.df + "/*_x.jpg")
        if len(self.files) == 0:
            raise FileNotFoundError("no files found in folder '" + self.df + "'")
        # retrieve image height and width
        self.imhw = cv2.imread(self.files[0]).shape[0]
        
    def __getitem__(self, index):
        # load image files
        x_file = self.files[index]
        y_file = self.df + "/" + x_file[len(self.df)+1:len(x_file)-6] + "_y.jpg"
        x = cv2.imread(x_file)
        y = cv2.imread(y_file)
        if y is None: #TODO investigate why this happens
            raise FileNotFoundError("unable to load '" + y_file + "'")
        if np.shape(x) != (self.imhw, self.imhw, 3):
            x = cv2.resize(x, (self.imhw, self.imhw), interpolation=cv2.INTER_LINEAR)
            y = cv2.resize(y, (self.imhw, self.imhw), interpolation=cv2.INTER_LINEAR)
        # RGB masks to classe masks
        y = np.abs(np.round(y/255)[:, :, :2] - (1, 0))
        # convert to tensors
        x = torch.from_numpy(x).float().permute(2, 0, 1)
        y = torch.from_numpy(y).float().permute(2, 0, 1)
        return x, y

    def __len__(self):
        return len(self.files)