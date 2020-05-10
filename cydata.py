import argparse 
import os
import sys
import logging
import json
import pandas as pd
import numpy as np
from cytomine import Cytomine
from cytomine.models import ImageInstance, Annotation
from glob import glob
import random
from shutil import copyfile
import cv2
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from shapely.geometry import box

def add_roi(id_project, id_image, hw=2048):
    if hw < 1:
        raise ValueError("the ROI height and width must be greater than 0")
    cred = json.load(open('credentials.json'))
    with Cytomine(host=cred['host'], 
                  public_key=cred['public_key'], 
                  private_key=cred['private_key']) as conn:
        rectangle = box(10, 10, hw, hw)
        annotation = Annotation(location=rectangle.wkt,
                                id_project=id_project,
                                id_image=id_image)
        annotation.save()

def download_dataset(filename, folder, imhw=512): #TODO download from ROI
    if imhw < 1:
        raise ValueError("the image height and width must be greater than 0")
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

def split_dataset(folder, split=0.8, seed=None):
    print("splitting dataset..")
    if (split < 0) or (split > 1):
        raise ValueError("split must be between 0 and 1")
    while folder[-1] == '/':
        folder = folder[:len(folder)-1]
    # load list of files
    files = glob(folder + "/*_x.jpg")
    if len(files) == 0:
        raise FileNotFoundError("no files found in folder '" + folder + "'")
    # shuffle the files
    if seed is not None:
        random.Random(seed).shuffle(files)
    else:
        random.shuffle(files)
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

def augment_dataset(folder):
    print("augmenting dataset..")
    # load list of files
    while folder[-1] == '/':
        folder = folder[:len(folder)-1]
    x_files = glob(folder + "/*_x.jpg")
    if len(x_files) == 0:
        raise FileNotFoundError("no files found in folder '" + folder + "'")
    # define transforms
    tf = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.TranslateX(percent=(-0.15, 0.15)),
        iaa.TranslateY(percent=(-0.15, 0.15)),
        iaa.Rot90((1, 4))
    ], random_order=True)

    for x_file in x_files:
        # load images
        file_id = x_file[len(folder)+1:len(x_file)-6]
        y_file = folder + "/" + file_id + "_y.jpg"
        x = cv2.imread(x_file)
        y = cv2.imread(y_file)
        if y is None:
            raise FileNotFoundError("unable to load '" + y_file + "'")
        if y.shape != x.shape:
            y = cv2.resize(y, (x.shape[1], x.shape[0]), cv2.INTER_LINEAR)
        # apply transforms
        y = SegmentationMapsOnImage(y, shape=x.shape)
        x, y = tf(image=x, segmentation_maps=y)
        # write files
        cv2.imwrite(folder + "/" + file_id + "a_x.jpg", x)
        cv2.imwrite(folder + "/" + file_id + "a_y.jpg", y.get_arr())
    print("dataset augmented")

def draw_borders(folder):
    print("drawing borders..")
    # load list of files
    while folder[-1] == '/':
        folder = folder[:len(folder)-1]
    x_files = glob(folder + "/*_x.jpg")
    if len(x_files) == 0:
        raise FileNotFoundError("no files found in folder '" + folder + "'")
    # create folder
    folder_b = folder + "_b"
    if not os.path.exists(folder_b):
        os.makedirs(folder_b)
    # add borders
    for x_file in x_files:
        # load files
        file_id = x_file[len(folder)+1:len(x_file)-6]
        y_file = folder + "/" + file_id + "_y.jpg"
        x = cv2.imread(x_file)
        y = cv2.imread(y_file)
        if y is None:
            raise FileNotFoundError("unable to load '" + y_file + "'")
        # compute borders
        kernel = np.ones((5, 5), np.uint8)
        erosion = cv2.erode(y, kernel, iterations = 2)
        borders = (y - erosion)*(1, 1, 0)
        y = y - borders
        # write files
        cv2.imwrite(folder_b + "/" + file_id + "_x.jpg", x)
        cv2.imwrite(folder_b + "/" + file_id + "_y.jpg", y)
    print("borders drawn")

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description='Cytomine datasets utilities.')
    parser.add_argument('-desc', metavar='filename', 
                        help='filename of the dataset descriptor')
    parser.add_argument('-hw', metavar='value', type=int,
                        help='image or ROI height and width')
    parser.add_argument('-split', metavar='value', type=float,
                        help='split value')
    parser.add_argument('-seed', metavar='value', type=int, help='seed value')
    parser.add_argument('-p', metavar='id', type=int, help="project id")
    parser.add_argument('-i', metavar='id', type=int, help="image id")
    parser.add_argument('m', metavar='mode',
                        help='modes: addroi, download, borders, split, augment')
    if 'addroi' not in sys.argv:
        parser.add_argument('df', metavar='folder', help='dataset directory')
    args = parser.parse_args()
    # run selected mode
    if args.m == 'addroi':
        if args.p is None:
            raise ValueError("must provide a project id")
        if args.i is None:
            raise ValueError("must provide an image id")
        if args.hw is not None:
            add_roi(args.p, args.i, args.hw)
        else:
            add_roi(args.p, args.i)
    elif args.m == 'download':
        if args.desc is None:
            raise ValueError("must provide a dataset descriptor")
        if args.hw is not None:
            download_dataset(args.desc, args.df, args.hw)
        else:
            download_dataset(args.desc, args.df)
    elif args.m == 'borders':
        draw_borders(args.df)
    elif args.m == 'split':
        if (args.split is not None) and (args.seed is not None):
            split_dataset(args.df, args.split, args.seed)
        elif args.split is not None:
            split_dataset(args.df, args.split)
        elif args.seed is not None:
            split_dataset(args.df, seed=args.seed)
        else:
            split_dataset(args.df)
    elif args.m == 'augment':
        augment_dataset(args.df)
    else:
        raise ValueError("unknown mode '" + args.m + "'")