import argparse 
from dataset import download_dataset, split_dataset, ImgSet
from unet import UnetSegmenter

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description='Cell segmentation in '
                                     + 'whole-slide cytological images '
                                     + 'of the thyroid.')
    parser.add_argument('-desc', metavar='filename', 
                        help='filename of the dataset descriptor')
    parser.add_argument('-imhw', metavar='number', type=int,
                        help='image height and width')
    parser.add_argument('-seed', metavar='number', type=int, help='seed value')
    parser.add_argument('-split', metavar='number', type=float,
                        help='split value')
    parser.add_argument('-model', metavar='filename', 
                        help='filename of the model to load/save')
    parser.add_argument('-epochs', metavar='number', type=int,
                        help='number of epochs for training')
    parser.add_argument('-psize', metavar='number', type=int, 
                        help='segmentation patch size')
    parser.add_argument('m', metavar='mode',
                        help='modes: download, split, train, segment')
    parser.add_argument('df', metavar='folder', help='directory of the dataset')
    args = parser.parse_args()
    # check mode
    modes = ['download', 'split', 'train', 'segment']
    if modes.__contains__(args.m) == False:
        raise ValueError("unknown mode '" + args.m + "'")        
    # run selected mode
    if args.m == 'download':
        if args.desc is None:
            raise ValueError("must provide a dataset descriptor")
        download_dataset(args.desc, args.df)
    if args.m == 'split':
        split_dataset(args.df)
    if args.m == 'train':
        segmenter = UnetSegmenter()
        if args.epochs:
            segmenter.train(ImgSet(args.df), args.epochs)
        else:
            segmenter.train(ImgSet(args.df), 3)
        if args.model:
            segmenter.save_model(args.model)
    if args.m == 'segment':
        segmenter = UnetSegmenter()
        if args.model:
            segmenter.load_model(args.model)
        if args.psize:
            segmenter.segment(ImgSet(args.df), args.psize)
        else:
            segmenter.segment(ImgSet(args.df))