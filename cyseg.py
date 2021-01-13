import argparse
from unet import UnetSegmenter, pred_pp


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', metavar='folder', help='dataset directory')
    parser.add_argument('-load', metavar='filename', 
                        help='filename of the model to load')
    parser.add_argument('-save', metavar='filename', 
                        help='filename of the model to save')
    parser.add_argument('-epochs', metavar='number', type=int, default=3,
                        help='number of epochs for training')
    parser.add_argument('-depth', metavar='value', type=int, default=16,
                        help='model initial depth')
    parser.add_argument('-classes', metavar='value', type=int, default=2,
                        help='number of classes')
    parser.add_argument('-weights', metavar='value', type=float, nargs='+',
                        help='class weights for loss computation')
    parser.add_argument('-tsize', metavar='value', type=int, default=512,
                        help='tile size')
    parser.add_argument('-thresh', metavar='value', type=float, default=0.5,
                        help='segmentation mask threshold')
    parser.add_argument('-dest', metavar='folder', default='',
                        help='segmentation destination folder')
    parser.add_argument('m', metavar='mode',
                        help='modes: train, segment')
    args = parser.parse_args()
    
    if args.m == 'train':
        if args.d is None:
            raise ValueError("must provide a dataset folder")
        
        # create segmenter
        segmenter = UnetSegmenter(args.classes, args.weights, args.depth)
        if args.load is not None:
            segmenter.load_model(args.load)

        # train
        segmenter.train(args.d, args.epochs, args.tsize)

        # save model
        if args.save is not None:
            segmenter.save_model(args.save)
    
    elif args.m == 'segment':
        if args.d is None:
            raise ValueError("must provide a dataset folder")
        
        # create segmenter
        segmenter = UnetSegmenter(args.classes, init_depth=args.depth)
        if args.load is not None:
            segmenter.load_model(args.load)
        
        # segment folder
        segmenter.segment(args.d, args.dest, args.tsize, pred_pp(args.thresh))
        
    else:
        raise ValueError("unknown mode '" + args.m + "'")
