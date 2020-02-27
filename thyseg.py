import sys
import argparse
from cytomine import Cytomine
from dataset import build_dataset

host = "https://research.cytomine.be"
public_key = 'b8a99025-7dfa-41af-b317-eb98c3c55302'
private_key = 'd7c53597-0de4-4255-b7cd-9e3db60bddc2'
project_id = 77150529

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description='Cell segmentation in whole-slide cytological images of the thyroid.')
    parser.add_argument('-f', metavar='filename', help='filename of the dataset descriptor')
    parser.add_argument('-m', metavar='modes', nargs='+', help='modes: dataset, train, segment')
    args = parser.parse_args()

    # check modes
    dataset = False
    train = False
    segment = False
    if(args.m is None):
       args.m = ['segment'] # default mode
    if(len(args.m) > 3):
        print("error: there must be between 1 and 3 different modes")
        exit(1)
    for mode in args.m:
        if(mode == 'dataset'):
            if(args.f is None):
                print("error: must provide a dataset descriptor")
                exit(1)
            if(not dataset):
                dataset = True
            else:
                print("error: duplicate of the mode 'dataset'")
                exit(1)
        elif(mode == 'train'):
            if(not train):
                train = True
            else:
                print("error: duplicate of the mode 'train'")
                exit(1)
        elif(mode == 'segment'):
            if(not segment):
                segment = True
            else:
                print("error: duplicate of the mode 'segment'")
                exit(1)
        else:
            print("error: unknown mode: " + mode)
            exit(1)
    
    if(dataset):
        print("building the dataset..")
        with Cytomine(host=host, public_key=public_key, private_key=private_key, verbose=None) as conn:
            build_dataset(args.f)
        print("dataset done")
    if(train):
        print("training the model..")
        print("training done")
    if(segment):
        print("segmenting..")
        print("segmentation done")