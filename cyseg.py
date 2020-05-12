import argparse
from segmenter import ImgSet
from unet import UnetSegmenter

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description='Cell segmentation in '
                                     + 'whole-slide cytological images.')
    parser.add_argument('-load', metavar='filename', 
                        help='filename of the model to load')
    parser.add_argument('-save', metavar='filename', 
                        help='filename of the model to save')
    parser.add_argument('-epochs', metavar='number', type=int,
                        help='number of epochs for training')
    parser.add_argument('-depth', metavar='value', type=int,
                        help='model initial depth')
    parser.add_argument('-thresh', metavar='value', type=float,
                        help='mask threshold value')
    parser.add_argument('-iters', metavar='number', type=int,
                        help='number of iterations in improve mode')
    parser.add_argument('-psize', metavar='value', type=int, 
                        help='segmentation patch size')
    parser.add_argument('-a', action='store_true',
                        help='flag for model assessment')
    parser.add_argument('m', metavar='mode',
                        help='modes: train, segment, improve')
    parser.add_argument('df', metavar='folder', help='dataset directory')
    args = parser.parse_args()
    # run selected mode
    if args.m == 'train':
        if args.depth is not None:
            segmenter = UnetSegmenter(args.depth)
        else:
            segmenter = UnetSegmenter()
        if args.load is not None:
            segmenter.load_model(args.load)
        if args.epochs is None:
            args.epochs = 3
        segmenter.train(ImgSet(args.df), args.epochs)
        if args.save is not None:
            segmenter.save_model(args.save)
    elif args.m == 'segment':
        segmenter = UnetSegmenter()
        if args.load is not None:
            segmenter.load_model(args.load)
        if args.psize is not None:
            segmenter.segment(ImgSet(args.df), psize=args.psize, 
                              thresh=args.thresh, assess=args.a)
        else:
            segmenter.segment(ImgSet(args.df), thresh=args.thresh, assess=args.a)
    elif args.m == 'improve':
        segmenter = UnetSegmenter()
        if args.load is not None:
            segmenter.load_model(args.load)
        if args.epochs is None:
            args.epochs = 1
        if args.iters is not None:
            if args.thresh is not None:
                segmenter.iter_data_imp(args.df, args.iters, args.epochs,
                                        args.thresh)
            else:
                segmenter.iter_data_imp(args.df, args.iters, args.epochs)
            if args.save is not None:
                segmenter.save_model(args.save)
        else:
            raise ValueError("must provide a number of iterations")
    else:
        raise ValueError("unknown mode '" + args.m + "'")