import argparse 
from dataset import download_dataset, ImgSet
from unet import UnetSegmenter

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description='Cell segmentation in '
                                     + 'whole-slide cytological images '
                                     + 'of the thyroid.')
    parser.add_argument('-d', metavar='filename', 
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
            if(args.d is None):
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
        download_dataset(args.d)
    if(train):
        segmenter = UnetSegmenter()
        segmenter.train(ImgSet('train'))
        segmenter.save_model('unet.pth')
    if(segment):
        segmenter = UnetSegmenter()
        segmenter.load_model('unet.pth')
        segmenter.segment(ImgSet('test'))