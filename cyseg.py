import sys
import argparse
import numpy as np
from unet import UnetSegmenter, UnetSegBuilder, seg_postprocess, idi_postprocess
from mpseg import mp_segment_wsi


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
    parser.add_argument('-i', metavar='id', type=int,
                        help='cytomine wsi id')
    parser.add_argument('-w', metavar='win', default=[],
                        help='wsi window in the form : [off_x,off_y,width,height]')
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
        
        # create segmenter
        segmenter = UnetSegmenter(init_depth=args.depth)
        if args.load is not None:
            segmenter.load_model(args.load)

        # train
        segmenter.train(args.d, args.epochs)

        # save model
        if args.save is not None:
            segmenter.save_model(args.save)
    
    elif args.m == 'segment':
        if (args.d is None) and (args.i is None):
            raise ValueError("must provide a dataset folder or an image identifier")
        
        # segment folder
        if args.d:
            # create segmenter
            segmenter = UnetSegmenter(init_depth=args.depth)
            if args.load is not None:
                segmenter.load_model(args.load)
            # segment
            segmenter.segment_folder(args.d, dest=args.dest, tsize=args.tsize,
                                     transform=seg_postprocess(args.thresh),
                                     assess=args.a)
        # segment WSI
        if args.i:
            # check window
            if args.w != []:
                try:
                    window = np.array(eval(args.w), dtype=np.int)
                except:
                    raise ValueError("invalid window: " + str(args.w))
                if window.shape != (4,):
                    raise ValueError("invalid window: " + str(args.w))
            else:
                window = np.array(args.w, dtype=np.int)

            # create Cytomine args dictionnary
            cy_args = {
                'host': args.host,
                'public_key': args.public_key,
                'private_key': args.private_key,
                'software_id': args.cytomine_id_software,
                'project_id': args.cytomine_id_project
            }
            
            if window != []:
                # create segmenter
                segmenter = UnetSegmenter(init_depth=args.depth)
                if args.load is not None:
                    segmenter.load_model(args.load)
                # compute polygons
                polygons = segmenter.segment_wsi(cy_args, args.i, window,
                                          tsize=args.tsize,
                                          transform=seg_postprocess(args.thresh))
            else:
                # create segmenter builder
                seg_builder = UnetSegBuilder(init_depth=args.depth,
                                             model_file=args.load)
                # compute polygons
                polygons = mp_segment_wsi(seg_builder, cy_args, args.i,
                                          w_width=15000, w_height=9000,
                                          tsize=args.tsize,
                                          transform=seg_postprocess(args.thresh))
            # upload annotations
            UnetSegmenter.upload_annotations_job(cy_args, args.i, polygons, window)
    
    elif args.m == 'improve':
        if args.d is None:
            raise ValueError("must provide a dataset folder")
        
        # create segmenter
        segmenter = UnetSegmenter(init_depth=args.depth)
        if args.load is not None:
            segmenter.load_model(args.load)
        
        # data improvement
        segmenter.iter_data_imp(args.d, args.iters, args.epochs,
                                transform=idi_postprocess(args.thresh))

        # save model
        if args.save is not None:
            segmenter.save_model(args.save)
    
    else:
        raise ValueError("unknown mode '" + args.m + "'")