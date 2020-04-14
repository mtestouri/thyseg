import argparse 
from dataset import download_dataset, ImgSet
from unet import UnetSegmenter

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description='Cell segmentation in '
                                     + 'whole-slide cytological images '
                                     + 'of the thyroid.')
    parser.add_argument('-dd', metavar='filename', 
                        help='filename of the dataset descriptor')
    parser.add_argument('-mf', metavar='filename', 
                        help='filename of the model to load/save')
    parser.add_argument('-ne', metavar='number', type=int,
                        help='number of epochs for training')
    parser.add_argument('m', metavar='modes', nargs='+', 
                        help='modes: dataset, train, segment')
    parser.add_argument('df', metavar='folder', help='directory of the dataset')
    args = parser.parse_args()
    # check modes
    dataset = False
    train = False
    segment = False
    if(len(args.m) > 3):
        raise ValueError("there must be between 1 and 3 different modes")
    for mode in args.m:
        if(mode == 'dataset'):
            if(args.dd is None):
                raise ValueError("must provide a dataset descriptor")
            if(not dataset):
                dataset = True
            else:
                raise ValueError("duplicate of the mode 'dataset'")
        elif(mode == 'train'):
            if(not train):
                train = True
            else:
                raise ValueError("duplicate of the mode 'train'")
        elif(mode == 'segment'):
            if(not segment):
                segment = True
            else:
                raise ValueError("duplicate of the mode 'segment'")
        else:
            raise ValueError("unknown mode '" + mode + "'")
    # run selected modes
    if(dataset):
        download_dataset(args.dd, args.df)
    if(train or segment):
        segmenter = UnetSegmenter()
        if(train):
            if args.ne:
                segmenter.train(ImgSet(args.df, 'train'), args.ne)
            else:
                segmenter.train(ImgSet(args.df, 'train'), 3)
            if args.mf:
                segmenter.save_model(args.mf)
        if(segment):
            if args.mf:
                segmenter.load_model(args.mf)
            segmenter.segment(ImgSet(args.df, 'test'))