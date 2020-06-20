import sys
import argparse
import numpy as np
from unet import UnetSegmenter, seg_postprocess, idi_postprocess
from cytomine import CytomineJob, Cytomine


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
    parser.add_argument('-depth', metavar='value', type=int, default=16,
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
                        help='wsi windows in the form : [off_x,off_y,width,height]'
                             + ' and the origin is the lower left corner')
    parser.add_argument('-cs', action='store_true',
                        help='flag to use CytomineSlide instead of OpenSlide')
    parser.add_argument('-a', action='store_true',
                        help='flag for model assessment')
    parser.add_argument('m', metavar='mode',
                        help='modes: train, segment, improve')
    # Cytomine arguments
    if "-i" in sys.argv:
        parser.add_argument('--host', required=True,
                            help='Cytomine host')
        parser.add_argument('--public_key', required=True,
                            help='Cytomine public key')
        parser.add_argument('--private_key', required=True,
                            help='Cytomine private key')
        parser.add_argument('--cytomine_id_project', required=True,
                            help='project id')
        parser.add_argument('--cytomine_id_software', required=True,
                            help='software id')
    args = parser.parse_args()
    
    if args.m == 'train':
        if args.d is None:
            raise ValueError("must provide a dataset folder")
        
        segmenter = UnetSegmenter(args.depth)
        if args.load is not None:
            segmenter.load_model(args.load)
        segmenter.train(args.d, args.epochs)
        if args.save is not None:
            segmenter.save_model(args.save)
    
    elif args.m == 'segment':
        if (args.d is None) and (args.i is None):
            raise ValueError("must provide a dataset folder or an image identifier")
        
        segmenter = UnetSegmenter(args.depth)
        if args.load is not None:
            segmenter.load_model(args.load)
        if args.d:
            segmenter.segment_folder(args.d, dest=args.dest, 
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

            # slide type
            if args.cs:
                slide_type = 'cytomineslide'
            else:
                slide_type = 'openslide'

            # create Cytomine context
            #with Cytomine(host=args.host,
            #              public_key=args.public_key,
            #              private_key=args.private_key) as conn:
            with CytomineJob(host=args.host,
                             public_key=args.public_key,
                             private_key=args.private_key,
                             software_id=args.cytomine_id_software,
                             project_id=args.cytomine_id_project) as job:
                
                # remote segmentation
                for image_id in args.i:
                    if len(windows) > 0:
                        for window in windows:
                            segmenter.segment_wsi(image_id, window, args.tsize,
                                        transform=seg_postprocess(args.thresh),
                                        slide_type=slide_type)
                    else:
                        segmenter.segment_wsi(image_id, [], args.tsize, 
                                        transform=seg_postprocess(args.thresh),
                                        slide_type=slide_type)
    
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