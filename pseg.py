from multiprocessing.pool import ThreadPool

import numpy as np
from unet import UnetSegmenter
from torch.utils.data import Dataset, DataLoader

from sldc import TileTopology, Tile, TileBuilder, SemanticMerger
from sldc_cytomine import CytomineSlide
from cytomine.models import ImageInstance, Annotation, AnnotationCollection


def p_segment_wsi(segmenter, image_id, w_width=15000, w_height=9000, tsize=512,
                  transform=None):
        """
        
        """

        # fetch wsi instance
        image_instance = ImageInstance().fetch(image_id)

        # create slide image and dataset
        wsi = CytomineSlide(image_id)
        dataset = WindowDataset(wsi, w_width, w_height, overlap=tsize)

        # inits
        count = 0
        window_polygons, window_ids = list(), list()

        # compute mask polygons
        dl = DataLoader(dataset=dataset, batch_size=1)
        for x, ids in dl:
            # window segmentation and collect merged polygons
            polygons = segmenter.segment_wsi(image_instance.id, x.n, 
                                           transform=transform)
            print(x)
            # TODO change ref of polygons
            # TODO check polygons shape
            window_polygons.append(polygons)

            window_ids.extend(ids.numpy())
            count += x.shape[0]
            print(f'processed windows {count}/{len(dataset)}')
        
        # merge polygons overlapping several tiles
        print("merging polygons..")
        merged = SemanticMerger(tolerance=1).merge(window_ids, window_polygons,
                                                   dataset.topology)
        
        return

        # upload to cytomine
        print("uploading annotations..")
        anns = AnnotationCollection()
        for polygon in merged:
            anns.append(
                Annotation(
                location=self._change_referential(polygon, 0, 0, 
                                                  image_instance.height).wkt,
                id_image=image_instance.id,
                id_project=image_instance.project,
                term=[FOREGROUND]
                )
            )
        anns.save(n_workers=4)


def change_referential(self, p, off_x, off_y, w_height):
    return affine_transform(p, [1, 0, 0, -1, off_x, off_y + w_height])


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
