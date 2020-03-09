import pandas as pd
import cv2
import numpy as np
from cytomine import Cytomine
from cytomine.models import ImageInstance, Annotation

from glob import glob
import random

X_SIZE = 512
SEED = 42
SPLIT = 0.7

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

                #image = cv2.imread("dataset/" + str(i) + "_x.jpg")
                #print(np.shape(image))
            
                slice_image.window(x, y, X_SIZE, X_SIZE, 
                                    dest_pattern="dataset/" + str(i) + "_y.jpg",
                                    mask=True,
                                    terms=annotation.term)
            i += 1
            if(i == 100):
                break

def load_train_set():
    # load dataset filenames
    files = glob("dataset/*_x.jpg")
    random.Random(SEED).shuffle(files)
    nb_files = len(files)
    # load files
    train_files = files[:round(SPLIT*nb_files)]
    x_train = []
    y_train = []
    for filename in train_files:
        x_train.append(cv2.imread(filename))
        y_train.append(cv2.imread("dataset/" + filename[8] + "_y.jpg"))
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    return [x_train, y_train]

def load_test_set():
    # load dataset filenames
    files = glob("dataset/*_x.jpg")
    random.Random(SEED).shuffle(files)
    nb_files = len(files)
    # load files
    test_files = files[round(SPLIT*nb_files):]
    x_test = []
    y_test = []
    for filename in test_files:
        x_test.append(cv2.imread(filename))
        y_test.append(cv2.imread("dataset/" + filename[8] + "_y.jpg"))
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    return [x_test, y_test]