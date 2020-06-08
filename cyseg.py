import sys
import argparse
import numpy as np
from segmenter import ImgSet
from unet import UnetSegmenter, seg_postprocess, idi_postprocess
from cytomine import CytomineJob, Cytomine
import json


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description='Cell segmentation in '
                                     + 'whole-slide cytological images.')
    parser.add_argument('-d', metavar='folder', help='dataset directory')
    parser.add_argument('-load', metavar='filename', 
                        help='filename of the model to load')
    parser.add_argument('-save', metavar='filename', 
                        help='filename of the model to save')
    parser.add_argument('-epochs', metavar='number', type=int, default=3,
                        help='number of epochs for training')
    parser.add_argument('-depth', metavar='value', type=int, default=32,
                        help='model initial depth')
    parser.add_argument('-thresh', metavar='value', type=float, default=0.5,
                        help='mask threshold value')
    parser.add_argument('-iters', metavar='number', type=int, default=2,
                        help='number of iterations in improve mode')
    parser.add_argument('-tsize', metavar='value', type=int, default=512,
                        help='segmentation tile size')
    parser.add_argument('-dest', metavar='folder', default='segmentations',
                        help='segmentations destination folder')
    parser.add_argument('-i', metavar='id', type=int, nargs='+',
                        help='cytomine wsi identifiers')
    parser.add_argument('-w', metavar='win', nargs='+', default=[],
                        help='wsi windows : [x,y,width,height ...]')
    parser.add_argument('-a', action='store_true',
                        help='flag for model assessment')
    parser.add_argument('m', metavar='mode',
                        help='modes: train, segment, improve')
    args = parser.parse_args()
    
    if args.m == 'train':
        if args.d is None:
            raise ValueError("must provide a dataset folder")
        
        segmenter = UnetSegmenter(args.depth)
        if args.load is not None:
            segmenter.load_model(args.load)
        segmenter.train(ImgSet(args.d), args.epochs)
        if args.save is not None:
            segmenter.save_model(args.save)
    
    elif args.m == 'segment':
        if (args.d is None) and (args.i is None):
            raise ValueError("must provide a dataset folder or an image identifier")
        
        segmenter = UnetSegmenter(args.depth)
        if args.load is not None:
            segmenter.load_model(args.load)
        if args.d:
            segmenter.segment(ImgSet(args.d, masks=args.a), dest=args.dest, 
                              tsize=args.tsize, assess=args.a, 
                              transform=seg_postprocess(args.thresh))
        if args.i:
            # check windows
            windows = list()
            for window in args.w:
                try:
                    window = np.array(eval(window), dtype=np.int)
                except:
                    raise ValueError("invalid window : " + str(window))
                if window.shape != (4,):
                    raise ValueError("invalid window : " + str(window))
                windows.append(window)
            windows = np.array(windows, dtype=np.int)

            cy = json.load(open('cytomine.json'))
            with Cytomine(host=cy['host'], 
                          public_key=cy['public_key'], 
                          private_key=cy['private_key']) as conn:
            #with CytomineJob(host=cy['host'],
            #                 public_key=cy['public_key'],
            #                 private_key=cy['private_key'],
            #                 software_id=cy['software_id'],
            #                 project_id=cy['project_id']) as job:
                segmenter.segment_r(args.i, windows, tsize=args.tsize,
                                    transform=seg_postprocess(args.thresh))
    
    elif args.m == 'improve':
        if args.d is None:
            raise ValueError("must provide a dataset folder")
        
        segmenter = UnetSegmenter(args.depth)
        if args.load is not None:
            segmenter.load_model(args.load)
        segmenter.iter_data_imp(args.d, args.iters, args.epochs,
                                transform=idi_postprocess(args.thresh))
        if args.save is not None:
            segmenter.save_model(args.save)
    
    else:
        raise ValueError("unknown mode '" + args.m + "'")