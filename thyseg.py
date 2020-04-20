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
    parser.add_argument('-split', metavar='number', type=float,
                        help='split value')
    parser.add_argument('-seed', metavar='number', type=int, help='seed value')
    parser.add_argument('-model', metavar='filename', 
                        help='filename of the model to load/save')
    parser.add_argument('-epochs', metavar='number', type=int,
                        help='number of epochs for training')
    parser.add_argument('-psize', metavar='number', type=int, 
                        help='segmentation patch size')
    parser.add_argument('-a', action='store_true',
                        help='flag for model assessment')
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
        if args.imhw is not None:
            download_dataset(args.desc, args.df, args.imhw)
        else:
            download_dataset(args.desc, args.df)
    if args.m == 'split':
        if (args.split is not None) and (args.seed is not None):
            split_dataset(args.df, args.split, args.seed)
        elif args.split is not None:
            split_dataset(args.df, args.split)
        elif args.seed is not None:
            split_dataset(args.df, seed=args.seed)
        else:
            split_dataset(args.df)
    if args.m == 'train':
        segmenter = UnetSegmenter()
        if args.epochs is None:
            args.epochs = 3
        segmenter.train(ImgSet(args.df), args.epochs)
        if args.model is not None:
            segmenter.save_model(args.model)
    if args.m == 'segment':
        segmenter = UnetSegmenter()
        if args.model is not None:
            segmenter.load_model(args.model)
        if args.psize is not None:
            segmenter.segment(ImgSet(args.df), args.psize, args.a)
        else:
            segmenter.segment(ImgSet(args.df), assess=args.a)