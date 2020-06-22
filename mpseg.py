from torch.multiprocessing import Process, set_start_method

import numpy as np
from torch.utils.data import Dataset, DataLoader

from cytomine import Cytomine
from cytomine.models import ImageInstance
from sldc import TileTopology, Tile, TileBuilder, SemanticMerger
from sldc_cytomine import CytomineSlide


def mp_segment_wsi(seg_builder, cy_args, image_id, w_width, w_height,
                   tsize=512, transform=None):
        """
        
        """

        # create Cytomine context
        with Cytomine(host=cy_args['host'],
                      public_key=cy_args['public_key'],
                      private_key=cy_args['private_key']) as conn:
            # fetch wsi instance
            image_instance = ImageInstance().fetch(image_id)
            # create slide image and dataset
            dataset = WindowDataset(CytomineSlide(image_id), w_width, w_height,
                                    overlap=tsize)

        # inits
        count = 0
        set_start_method('spawn')
        window_polygons, window_ids = list(), list()

        # compute mask polygons
        dl = DataLoader(dataset=dataset, batch_size=1)
        for x, ids in dl:
            # compute polygons in the window using a process
            print(x.squeeze(0).numpy())
            p = Process(target=_worker, args=(seg_builder, cy_args, 
                                              image_instance.id,
                                              x.squeeze(0).numpy(), 
                                              tsize, transform))
            p.start()
            p.join()
            
            # TODO change ref of polygons
            # TODO check polygons shape
            #window_polygons.append(polygons)
            
            #window_ids.extend(ids.numpy())
            count += x.shape[0]
            print(f'processed windows {count}/{len(dataset)}')
        
        return []

        # merge polygons overlapping several tiles
        print("merging polygons..")
        merged = SemanticMerger(tolerance=1).merge(window_ids, window_polygons,
                                                   dataset.topology)
        return merged


def _worker(seg_builder, cy_args, image_id, window, tsize, transform):
    # create segmenter
    segmenter = seg_builder.build()
    # compute polygons
    polygons = segmenter.segment_wsi(cy_args, image_id, window, tsize,
                                     transform=transform)


def _change_referential(p, off_x, off_y, w_height):
    return affine_transform(p, [1, 0, 0, -1, off_x, off_y + w_height])


#def _skip_tile(tile_id, topology):
#    tile_row, tile_col = topology._tile_coord(tile_id)
#    skip_bottom = (topology._image.height 
#                   % (topology._max_height - topology._overlap)) != 0
#    skip_right = (topology._image.width 
#                  % (topology._max_width - topology._overlap)) != 0
#    return (skip_bottom and tile_row == topology.tile_vertical_count - 1) or \
#           (skip_right and tile_col == topology.tile_horizontal_count - 1)


class WindowDataset(Dataset):
    """
    SLDC window dataset

    parameters
    ----------
    wsi: Image
        CytomineSlide object
        
    w_width: int
        window width

    w_height: int
        window height

    overlap: int
        overlap between windows
    """
    
    def __init__(self, wsi, w_width, w_height, overlap):
        self._wsi = wsi
        self._topology = TileTopology(
            image=wsi, tile_builder=WindowBuilder(),
            max_width=w_width, max_height=w_height,
            overlap=overlap
        )
        self._w_width = w_width
        self._w_height = w_height
        self._overlap = overlap

    @property
    def topology(self):
        return self._topology

    def __getitem__(self, index):
        """
        returns (window features, window identifier)
        """

        identifier = index + 1
        window = self._topology.tile(identifier).np_image
        
        # maybe TODO right and bottom padding

        return window, identifier

    def __len__(self):
        return self.topology.tile_count


class Window(Tile):
    """
    Special type of Tile object representing a window

    This object is used to take advantage of the topology and merging features 
    of SLDC without effectively loading the window in memory and, instead, 
    by loading numpy array representing the window location and size
    """

    @property
    def np_image(self):
        """
        returns window features in the form [offset_x, offset_y, width, height]
        """
        return np.array([self.offset_x, self.offset_y, self.width, self.height])


class WindowBuilder(TileBuilder):
    """
    Window object builder
    """

    def build(self, image, offset, width, height, polygon_mask=None):
        return Window(image, offset, width, height)
