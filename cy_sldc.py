import os
import sys

import numpy as np
from argparse import ArgumentParser

from cytomine import CytomineJob
from cytomine.models import ImageInstance, Annotation
from shapely.affinity import affine_transform

from sldc import TileTopology, SemanticLocator, SemanticMerger
from sldc_openslide import OpenSlideImage, OpenSlideTileBuilder
from torch.utils.data import Dataset

from unet import UnetSegmenter

def change_referential(p, height):
    return affine_transform(p, [1, 0, 0, -1, 0, height])


def skip_tile(tile_id, topology):
    tile_col, tile_row = topology._tile_coord(tile_id)
    skip_bottom = (topology._image.height % topology._max_height) != 0
    skip_right = (topology._image.width % topology._max_width) != 0
    return (skip_bottom and tile_row == topology.tile_vertical_count - 1) or \
           (skip_right and tile_col == topology.tile_horizontal_count - 1)


class SldcDataset(Dataset):
    """for inference"""
    def __init__(self, wsi, tile_width, tile_height, overlap, skip_border=True):
        """
        :param skip_border: True for skipping border tiles of which dimensions does
        not match (tile_height, tile_width)
        """
        self._wsi = wsi
        topology = TileTopology(
            image=wsi, tile_builder=OpenSlideTileBuilder(),
            max_width=tile_width, max_height=tile_height,
            overlap=overlap
        )
        self._topology = topology
        self._skip_border = skip_border
        self._tile_width = tile_width
        self._tile_height = tile_height
        self._overlap = overlap
        # maps dataset index with tile identifier
        if not skip_border:
            self._kept_tile_map = np.arange(self._topology.tile_count) + 1
        else:
            self._kept_tile_map = np.where([not skip_tile(tile.identifier, topology) for tile in topology])[0] + 1

    @property
    def topology(self):
        return self._topology

    def __getitem__(self, index):
        """Return (numpy image of a tile, tile identifier)"""
        identifier = self._kept_tile_map[index]
        tile = self._topology.tile(identifier)
        return tile.np_image, identifier

    def __len__(self):
        return self._kept_tile_map.shape[0]


def main(argv):
    argparser = ArgumentParser()
    argparser.add_argument("-i", "--image", dest="id_image", type=int, help="Cytomine id of the image to preocess")
    argparser.add_argument("-w", "--working_path", dest="working_path", help="Cytomine id of the image to preocess")
    argparser.add_argument("-t", "--threshold", dest="threshold", type=float, default=0.5)
    argparser.add_argument("-d", "--tile_dim", dest="tile_dim", type=int, default=512)
    argparser.add_argument("-o", "--overlap", dest="overlap", type=int, default=256)
    params, _ = argparser.parse_known_args(argv)

    with CytomineJob.from_cli(argv) as job:
        image_instance = ImageInstance().fetch(params.id_image)
        filepath = os.path.join(params.working_path, image_instance.originalFilename)
        
        os.makedirs(params.working_path, exist_ok=True)

        if not image_instance.download(filepath, override=False):
            raise ValueError("wsi image not downloaded")

        wsi = OpenSlideImage(filepath)
        dataset = SldcDataset(wsi, params.tile_dim, params.tile_dim, params.overlap, skip_border=True)

        w = wsi.window()
        # ---------------------------------------
        # plug your model and dataloader here
        # /!\ model in no_grad and eval modes
        segmenter = UnetSegmenter()
        loader = DataLoader(dataset=dataset, batch_size=4, num_workers=2)

        polygons, tile_ids = list(), list()
        for x, ids in loader:
            # x should be a batch of tiles
            # predict y, turn into classes by thresholding
            y = segmenter.predict(x)#net.forward(x)
            
            #TODO convert to 1 channel

            # insert post processing here (if any)
            y_cls = y > params.threshold

            # turn prediction into polygons
            locator = SemanticLocator(background=0)  # transform mask to polygons
            batch_size = x.dims(0)
            for i in range(batch_size):
                mask = y[i].cpu().numpy()
                polygons.append(locator.locate(mask, offset=dataset.topology.tile_offset(ids[i])))
            tile_ids.extend(ids)
        # ---------------------------------------

        # merge polygon overlapping several tiles
        merged = SemanticMerger(tolerance=0).merge(tile_ids, polygons, dataset.topology)

        # upload to cytomine
        for polygon in merged:
            Annotation(
                location=change_referential(polygon, wsi.height).wkt,
                id_image=image_instance.id,
                id_project=image_instance.id_project
            ).save()

if __name__ == "__main__":
    main(sys.argv[1:])