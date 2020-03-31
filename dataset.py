import pandas as pd
import numpy as np
from cytomine import Cytomine
from cytomine.models import ImageInstance, Annotation
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
project_id = 77150529

def download_dataset(filename):
    with Cytomine(host=host, public_key=public_key, private_key=private_key,
                  verbose=None) as conn:
        # dataset descriptor
        array = pd.read_csv(filename, sep=';').to_numpy()
        # download the images
        i = 0
        for row in array:
            if(i > 0):
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
                slice_image.window(x, y, IMG_HW, IMG_HW, 
                                    dest_pattern="dataset/" + str(i) + "_x.jpg")
                slice_image.window(x, y, IMG_HW, IMG_HW, 
                                    dest_pattern="dataset/" + str(i) + "_y.jpg",
                                    mask=True,
                                    terms=annotation.term)
            i += 1

def load_dataset(type):
    # load dataset filenames
    files = glob("dataset/*_x.jpg")
    random.Random(SEED).shuffle(files)
    if type == 'train':
        files = files[:round(SPLIT*len(files))]
    elif type == 'test':
        files = files[round(SPLIT*len(files)):]
    else:
        print("error: invalid dataset type '" + type + "'")
        exit(1)
    # load files
    x = np.empty((len(files), IMG_HW, IMG_HW, 3), dtype=np.float32)
    y = np.empty((len(files), IMG_HW, IMG_HW, 3), dtype=np.float32)
    i = 0
    for filename in files:
        x_img = cv2.imread(filename)
        y_img = cv2.imread("dataset/" + filename[8:len(filename)-6] + "_y.jpg")
        if(np.shape(x_img) != (IMG_HW, IMG_HW, 3)):
            x_img = cv2.resize(x_img, (IMG_HW, IMG_HW), interpolation=cv2.INTER_LINEAR)
            y_img = cv2.resize(y_img, (IMG_HW, IMG_HW), interpolation=cv2.INTER_LINEAR)
        x[i] = x_img
        y[i] = y_img
        i += 1
    # convert RGB masks to classe masks
    y = np.abs(np.round(y/255)[:, :, :, :2] - (1, 0))
    #for i in range(len(x)):
    #    for j in range(len(x[i])):
    #        for k in range(len(x[i, j])):
    #            print(x[i, j, k])
    #for i in range(len(y)):
    #    for j in range(len(y[i])):
    #        for k in range(len(y[i, j])):
    #            print(y[i, j, k])
    return [x, y]
