import argparse 
from dataset import download_dataset, load_dataset
from unet import UnetSegmenter

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description='Cell segmentation in '
                                     + 'whole-slide cytological images '
                                     + 'of the thyroid.')
    parser.add_argument('-f', metavar='filename', 
                        help='filename of the dataset descriptor')
    parser.add_argument('m', metavar='modes', nargs='+', 
                        help='modes: dataset, train, segment')
    args = parser.parse_args()
    # check modes
    dataset = False
    train = False
    segment = False
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
    # run selected modes
    if(dataset):
        print("downloading dataset..")
        download_dataset(args.f)
        print("dataset done")
    if(train):
        print("training the model..")
        (x_train, y_train) = load_dataset('train')
        UnetSegmenter().train(x_train, y_train, 'unet.pth')
        print("training done")
    if(segment):
        print("segmenting..")
        (x_test, y_test) = load_dataset('test')
        UnetSegmenter().segment(x_test, y_test, 'unet.pth')
        print("segmentation done")