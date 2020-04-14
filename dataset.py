import os
import sys
import logging
import pandas as pd
import numpy as np
from cytomine import Cytomine
from cytomine.models import ImageInstance, Annotation
import torch
from torch.utils.data import Dataset
from glob import glob
import random
import cv2

# dataset parameters
SEED = 42
IMG_HW = 512
SPLIT = 0.8
# cytomine connection
host = "https://research.cytomine.be"
public_key = 'b8a99025-7dfa-41af-b317-eb98c3c55302'
private_key = 'd7c53597-0de4-4255-b7cd-9e3db60bddc2'
logger = logging.getLogger()
logger.disabled = True

def download_dataset(filename, folder):
    print("downloading dataset..")
    with Cytomine(host=host, 
                  public_key=public_key, 
                  private_key=private_key) as conn:
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
            x = round(float(row[3])-(IMG_HW/2))
            y = image.height - round(float(row[4])+(IMG_HW/2))
            # download slice and corresponding mask 
            slice_image = image.reference_slice()
            slice_image.window(x, y, IMG_HW, IMG_HW, dest_pattern=folder
                               + "/" + str(i) + "_x.jpg")
            slice_image.window(x, y, IMG_HW, IMG_HW, dest_pattern=folder 
                               + "/" + str(i) + "_y.jpg",
                               mask=True,
                               terms=annotation.term)
            if i > 1:
                sys.stdout.write("\033[F")
            sys.stdout.write("\033[K")
            print(f'progress: {round(i/len(array)*100, 1)}%')
            i += 1
    print("dataset downloaded")

class ImgSet(Dataset):
    def __init__(self, folder, type):
        self.df = folder
        # load dataset filenames
        files = glob(self.df + "/*_x.jpg")
        if len(files) == 0:
            print("error: no files found in folder '" + folder + "'")
            exit(1)
        # shuffle the files
        random.Random(SEED).shuffle(files)
        if type == 'train':
            self.files = files[:round(SPLIT*len(files))]
        elif type == 'test':
            self.files = files[round(SPLIT*len(files)):]
        else:
            print("error: invalid dataset type '" + type + "'")
            exit(1)
        
    def __getitem__(self, index):
        # load image files
        x_file = self.files[index]
        y_file = self.df + "/" + x_file[len(self.df):len(x_file)-6] + "_y.jpg"
        x = cv2.imread(x_file)
        y = cv2.imread(y_file)
        if y is None: #TODO investigate why this happens
            raise Exception("error: unable to load '" + y_file + "'")
        if(np.shape(x) != (IMG_HW, IMG_HW, 3)):
            x = cv2.resize(x, (IMG_HW, IMG_HW), interpolation=cv2.INTER_LINEAR)
            y = cv2.resize(y, (IMG_HW, IMG_HW), interpolation=cv2.INTER_LINEAR)
        # RGB masks to classe masks
        y = np.abs(np.round(y/255)[:, :, :2] - (1, 0))
        # convert to tensors
        x = torch.from_numpy(x).float().permute(2, 0, 1)
        y = torch.from_numpy(y).float().permute(2, 0, 1)
        return x, y

    def __len__(self):
        return len(self.files)