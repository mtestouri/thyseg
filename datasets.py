import os, re
import numpy as np
import torch, cv2
from torch.utils.data import Dataset
from torch.nn.functional import relu
from sldc import TileTopology
from sldc_openslide import OpenSlideImage, OpenSlideTileBuilder 


class ImgDataset(Dataset):
    """
    image dataset
    
    parameters
    ----------
    folder: string
        folder of the dataset to load, files are in the form 
        'id_x.ext' (images) and 'id_y_channel.ext' (masks)
    """

    def __init__(self, folder):
        while folder[-1] == '/':
            folder = folder[:len(folder)-1]
        self._df = folder

        # load list of image files
        image_regex = r'\d*_x\.'
        self._files = [f for f in os.listdir(self._df) if re.search(image_regex, f)]
        if len(self._files) == 0:
            raise FileNotFoundError("no files found in folder '" + self._df + "'")
    
    def __getitem__(self, index):
        x_file = self._files[index]

        # search for the mask
        mask_regex = re.escape(x_file[:len(x_file)-6]) + r'_y_\d*\.'    
        y_files = [f for f in os.listdir(self._df) if re.search(mask_regex, f)]
        mask = []
        for y_file in y_files:
            mask.append(OpenSlideImage(self._df + "/" + y_file))

        return OpenSlideImage(self._df + "/" + x_file), mask

    def __len__(self):
        return len(self._files)


class TileDataset(Dataset):
    """
    tile dataset

    parameters
    ----------
    image: Image
        OpenSlideImage object
        
    mask: List
        list of OpenSlideImage objects representing the mask channels
    
    mask_merge: boolean
        toggle the merging of the masks into a single class

    tsize: int
        tile size

    overlap: int
        overlap between tiles
    """
    
    def __init__(self, image, mask=[], mask_merge=False, tsize=512, overlap=128):
        self._image_topology = TileTopology(
            image=image, tile_builder=OpenSlideTileBuilder(),
            max_width=tsize, max_height=tsize, overlap=overlap
        )

        self._mask_topology = []
        for channel in mask:
            self._mask_topology.append(TileTopology(
                image=channel, tile_builder=OpenSlideTileBuilder(),
                max_width=tsize, max_height=tsize, overlap=overlap
            ))

        self._tsize = tsize
        self._overlap = overlap
        self._mask_merge = mask_merge

    @property
    def topology(self):
        return self._image_topology

    def _check_padding(self, tile):
        # right padding
        if tile.shape[0] != self._tsize:
            pad = np.zeros((self._tsize - tile.shape[0], 
                            tile.shape[1], tile.shape[2]), dtype=np.uint8)
            tile = np.concatenate((tile, pad), axis=0)
        # bottom padding
        if tile.shape[1] != self._tsize:
            pad = np.zeros((tile.shape[0], self._tsize - tile.shape[1], 
                            tile.shape[2]), dtype=np.uint8)
            tile = np.concatenate((tile, pad), axis=1)
        return tile

    def __getitem__(self, index):
        identifier = index + 1
        
        # extract image tile
        image_tile = self._image_topology.tile(identifier).np_image
        # remove alpha channel and convert to RGB
        image_tile = cv2.cvtColor(image_tile, cv2.COLOR_BGRA2RGB)
        # convert to tensor
        image_tile = torch.from_numpy(self._check_padding(image_tile)).permute(2, 0, 1)

        mask_tile = None
        for i in range(len(self._mask_topology)):
            # extract mask tile
            tile = self._mask_topology[i].tile(identifier).np_image
            # remove alpha channel and convert to binary mask
            tile = cv2.cvtColor(tile, cv2.COLOR_BGRA2GRAY)
            _, tile = cv2.threshold(tile, 1, 1, cv2.THRESH_BINARY)
            tile = np.expand_dims(tile, 2)
            # convert to tensor
            tile = torch.from_numpy(self._check_padding(tile)).permute(2, 0, 1)

            if i == 0:
                mask_tile = torch.ones((1, tile.shape[1], tile.shape[2]), dtype=torch.int)
                mask_tile = torch.cat((mask_tile, tile), dim=0)
            # update background channel
            mask_tile[0,:,:] = relu(mask_tile[0,:,:] - tile)

            if self._mask_merge:
                mask_tile[1,:,:] = torch.bitwise_or(mask_tile[1,:,:], tile)
            elif i != 0:
                mask_tile = torch.cat((mask_tile, tile), dim=0)

        return image_tile.float(), mask_tile.float(), identifier

    def __len__(self):
        return self.topology.tile_count