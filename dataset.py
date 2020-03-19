import pandas as pd
import cv2
import numpy as np
from cytomine import Cytomine
from cytomine.models import ImageInstance, Annotation

from glob import glob
import random

X_SIZE = 512
SEED = 42
SPLIT = 0.95

# cytomine connection
host = "https://research.cytomine.be"
public_key = 'b8a99025-7dfa-41af-b317-eb98c3c55302'
private_key = 'd7c53597-0de4-4255-b7cd-9e3db60bddc2'
project_id = 77150529

def download_dataset(filename):
    with Cytomine(host=host, public_key=public_key, private_key=private_key, verbose=None) as conn:
        array = pd.read_csv(filename, sep=';').to_numpy()

        i = 0
        for row in array:
            if(i > 0):
                annotation = Annotation()
                annotation.id = int(row[0])
                annotation.fetch()

                image = ImageInstance()
                image.id = int(row[5])
                image.fetch()
            
                x = round(float(row[3])-(X_SIZE/2))
                y = image.height - round(float(row[4])+(X_SIZE/2))
                slice_image = image.reference_slice()
                slice_image.window(x, y, X_SIZE, X_SIZE, 
                                    dest_pattern="dataset/" + str(i) + "_x.jpg")
                slice_image.window(x, y, X_SIZE, X_SIZE, 
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
    x = np.empty((len(files), X_SIZE, X_SIZE, 3), dtype=np.float32)
    y = np.empty((len(files), X_SIZE, X_SIZE, 3), dtype=np.float32)
    i = 0
    for filename in files:
        x_img = cv2.imread(filename)
        y_img = cv2.imread("dataset/" + filename[8:len(filename)-6] + "_y.jpg")
        if(np.shape(x_img) != (X_SIZE, X_SIZE, 3)):
            x_img = cv2.resize(x_img, (X_SIZE, X_SIZE), interpolation=cv2.INTER_LINEAR)
            y_img = cv2.resize(y_img, (X_SIZE, X_SIZE), interpolation=cv2.INTER_LINEAR)
        x[i] = x_img
        y[i] = y_img
        i += 1
    # create classes
    y = y/255 # normalize
    y = y*[1, 1, 0] + [-1, 0, 0]
    y = np.abs(y)
    
    #for i in range(len(y)):
    #    for j in range(len(y[i])):
    #        for k in range(len(y[i, j])):
    #            print(y[i, j, k])

    return [x, y]
