import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
import cv2
from glob import glob
from shutil import copyfile
from metrics import dice, jaccard
from transforms import Resize

from cytomine.models import ImageInstance, Annotation
from shapely.affinity import affine_transform
from sldc import TileTopology, SemanticLocator, SemanticMerger
from sldc_cytomine import CytomineSlide, CytomineTileBuilder

class Segmenter:
    def __init__(self, c_weights=None):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.model = None
        self.c_weights = c_weights

    def check_model_init(self):
        if self.model is None:
            raise ValueError("variable 'self.model' not initialized")

    def save_model(self, model_file):
        self.check_model_init()
        torch.save(self.model.state_dict(), model_file)

    def load_model(self, model_file):
        """ Load the weights from a model file
        
        Beware that the depths of the segmenter and the model file 
        must match.
        """
        self.check_model_init()
        self.model.load_state_dict(torch.load(model_file))

    def set_eval():
        self.model.eval()

    def set_train():
        self.model.train()

    def train(self, dataset, n_epochs):
        raise NotImplementedError

    def predict(self, images, transform=None):
        self.check_model_init()
        train_mode = self.model.training
        if train_mode:
            self.model.eval()
        # compute masks
        with torch.no_grad():
            masks = torch.sigmoid(self.model(images.to(self.device)))
        # post-processing
        if transform is not None:
            for i in range(len(masks)):
                masks[i] = transform(masks[i].cpu()).to(self.device)
        if train_mode:
            self.model.train()
        return masks

    def segment(self, dataset, dest='segmentations', tsize=None, batch_size=1,  
                transform=None, assess=False):
        print("segmenting..")
        tsize_given = (tsize is not None)
        if tsize_given:
            if tsize < 1:
                raise ValueError("'tsize' must be greater than 0")
            else:
                tile_h = tsize
                tile_w = tsize
        # create folder
        while dest[-1] == '/':
            dest = dest[:len(dest)-1]
        if not os.path.exists(dest):
            os.makedirs(dest)
        
        if assess:
            sum_dice = 0
            sum_jaccard = 0
        self.model.eval()
        tf_resize = Resize()
        dl = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=2)
        for i, (images, masks, files_id) in enumerate(dl):
            # check dimensions
            im_h = images[0].shape[1]
            im_w = images[0].shape[2]
            if not tsize_given:
                tile_h = im_h
                tile_w = im_w
            if (im_h % tile_h) != 0 or (im_w % tile_w) != 0:
                raise ValueError("the tile size must divide the image dimensions")
            
            # compute masks using tiles
            masks_p = torch.zeros((batch_size, 2, im_h, im_w), dtype=torch.float32)
            i_h = 0
            for j in range(im_h // tile_h):
                i_w = 0
                for k in range(im_w // tile_w):
                    tiles = images[:, :, i_h:(i_h+tile_h), i_w:(i_w+tile_w)]
                    preds = self.predict(tiles)
                    
                    for l in range(len(preds)):
                        mask = preds[l].cpu()
                        # resize if necessary
                        if mask.shape != (2, tile_h, tile_w):
                            mask = tf_resize(mask, (tile_w, tile_h))
                        # write mask tile
                        masks_p[l, :, i_h:(i_h+tile_h), i_w:(i_w+tile_w)] = mask
                    i_w += tile_w
                i_h += tile_h
            
            for j in range(len(images)):
                image = images[j]
                mask = masks[j]
                mask_p = masks_p[j]
                # post-processing
                if transform is not None:
                    mask_p = transform(mask_p)

                if assess:
                    # metrics
                    mask = mask.unsqueeze(0)
                    mask_p = mask_p.unsqueeze(0)
                    sum_dice += dice(mask_p, mask, self.c_weights).item()
                    sum_jaccard += jaccard(mask_p, mask, self.c_weights).item()
                    mask = mask.squeeze(0)
                    mask_p = mask_p.squeeze(0)
                    # convert tensors to numpy
                    image = image.permute(1, 2, 0).cpu().numpy()
                    mask = mask.permute(1, 2, 0).numpy()
                    mask_p = mask_p.permute(1, 2, 0).numpy()
                    # select foreground channel
                    mask = mask[:, :, 1].reshape(im_h, im_w, 1)*180
                    mask_p = mask_p[:, :, 1].reshape(im_h, im_w, 1)*180
                    # convert to RGB image
                    z = np.zeros((im_h, im_w, 1), dtype=np.float32)
                    mask = np.concatenate((z, mask, z), axis=2)
                    mask_p = np.concatenate((z, mask_p, z), axis=2)
                    # combine masks and images
                    sup = image + mask
                    sup_p = image + mask_p
                    # write final image
                    sep = np.ones((im_h, 10, 3), dtype=np.float32)*255
                    img = np.concatenate((image, sep, 
                                          sup, sep, 
                                          sup_p, sep, 
                                          mask, sep, 
                                          mask_p), axis=1)
                    cv2.imwrite(dest + "/seg" + str(i+1) + ".jpg", img)
                else:
                    # convert tensor to numpy
                    mask_p = mask_p.permute(1, 2, 0).numpy()
                    # select foreground channel
                    mask_p = mask_p[:, :, 1].reshape(im_h, im_w, 1)*255
                    # convert to RGB image
                    mask_p = np.concatenate((mask_p, mask_p, mask_p), axis=2)
                    # write final image
                    cv2.imwrite(dest  + "/" + files_id[j] + "_y.jpg", mask_p)

                # display info
                if assess:
                    print("segmentation: " + str(i+1) + "/" + str(len(dataset))
                          + ", avg_dice: " + str(round(sum_dice/(i+1), 4))
                          + ", avg_jaccard: " + str(round(sum_jaccard/(i+1), 4))
                          , end='\r')
                else:
                    print(f'segmentation: {(i+1)}/{len(dataset)}', end='\r')
        print("\nsegmentation done")

    def iter_data_imp(self, folder, n_iters, n_epochs, transform=None):
        # load the file list
        while folder[-1] == '/':
            folder = folder[:len(folder)-1]
        x_files = glob(folder + "/*_x.jpg")
        y_files = glob(folder + "/*_y.jpg")
        if len(x_files) == 0:
            raise FileNotFoundError("no files found in folder '" + folder + "'")
        # copy the files into the new folder
        dest = folder + "_imp"
        if not os.path.exists(dest):
            os.makedirs(dest)
        for x_file in x_files:
            copyfile(x_file, dest + "/" + x_file[len(folder)+1:])
        for y_file in y_files:
            copyfile(y_file, dest + "/" + y_file[len(folder)+1:])
        # improve the data
        for i in range(n_iters):
            self.train(ImgSet(dest), n_epochs)
            self.segment(ImgSet(dest), dest=dest, transform=transform)
            
            # merge ground truth masks with predicted masks
            for y_file in y_files:
                y = cv2.imread(y_file)
                y_pred = cv2.imread(dest + "/" + y_file[len(folder)+1:])

                if y.shape != y_pred.shape:
                    y_pred = cv2.resize(y_pred, (y.shape[1], y.shape[0]),
                                        cv2.INTER_LINEAR)
                
                y_pred = cv2.bitwise_or(y, y_pred)
                cv2.imwrite(dest  + "/" + y_file[len(folder)+1:], y_pred)

    def segment_r(self, images, tsize, batch_size=4, transform=None, assess=False):
        overlap = int(round(tsize / 2))
        #TODO boucler sur les images SLDC
        for image in images:
            # fetch image and build tile dataset
            image_instance = ImageInstance().fetch(image)
            wsi = CytomineSlide(image) #w = wsi.window()
            dataset = SldcDataset(wsi, tsize, tsize, overlap, skip_border=True)

            polygons, tile_ids = list(), list()
            dl = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=2)
            for x, ids in dl:
                y = self.predict(x, transform) # predict tile masks
                y = y[:, 1, :, :] # convert to 1 channel

                # turn prediction into polygons
                locator = SemanticLocator(background=0) # transform mask to polygons
                batch_size = x.dims(0)
                for i in range(batch_size):
                    mask = y[i].cpu().numpy()
                    polygons.append(locator.locate(mask, 
                                    offset=dataset.topology.tile_offset(ids[i])))
                tile_ids.extend(ids)
            
            # merge polygon overlapping several tiles
            merged = SemanticMerger(tolerance=0).merge(tile_ids, polygons,
                                                       dataset.topology)
            # upload to cytomine
            for polygon in merged:
                Annotation(
                    location=change_referential(polygon, wsi.height).wkt,
                    id_image=image_instance.id,
                    id_project=image_instance.id_project
                ).save()

def change_referential(p, height):
    return affine_transform(p, [1, 0, 0, -1, 0, height])

def skip_tile(tile_id, topology):
    tile_col, tile_row = topology._tile_coord(tile_id)
    skip_bottom = (topology._image.height % topology._max_height) != 0
    skip_right = (topology._image.width % topology._max_width) != 0
    return (skip_bottom and tile_row == topology.tile_vertical_count - 1) or \
           (skip_right and tile_col == topology.tile_horizontal_count - 1)

class ImgSet(Dataset):
    def __init__(self, folder):
        while folder[-1] == '/':
            folder = folder[:len(folder)-1]
        self.df = folder
        # load list of files
        self.files = glob(self.df + "/*_x.jpg")
        if len(self.files) == 0:
            raise FileNotFoundError("no files found in folder '" + self.df + "'")
        # define dataset image size
        self.im_h = None
        self.im_w = None
        (x, _, _) = self.__getitem__(0)
        self.im_h = x.shape[1]
        self.im_w = x.shape[2]
        
    def __getitem__(self, index):
        # load image files
        x_file = self.files[index]
        file_id = x_file[len(self.df)+1:len(x_file)-6]
        y_file = self.df + "/" + file_id + "_y.jpg"
        x = cv2.imread(x_file)
        y = cv2.imread(y_file)
        if y is None:
            raise FileNotFoundError("unable to load '" + y_file + "'")
        # check size
        if (self.im_h is not None) and (self.im_w is not None):
            if (x.shape[0] != self.im_h) or (x.shape[1] != self.im_w):
                x = cv2.resize(x, (self.im_w, self.im_h), cv2.INTER_LINEAR)
            if (y.shape[0] != self.im_h) or (y.shape[1] != self.im_w):
                y = cv2.resize(y, (self.im_w, self.im_h), cv2.INTER_LINEAR) 
        # RGB masks to classe masks
        y = np.abs(np.round(y/255)[:, :, :2] - (1, 0))
        # convert to tensors
        x = torch.from_numpy(x).float().permute(2, 0, 1)
        y = torch.from_numpy(y).float().permute(2, 0, 1)
        return x, y, file_id

    def __len__(self):
        return len(self.files)

class SldcDataset(Dataset):
    """for inference"""
    def __init__(self, wsi, tile_width, tile_height, overlap, skip_border=True):
        """
        :param skip_border: True for skipping border tiles of which dimensions does
        not match (tile_height, tile_width)
        """
        self._wsi = wsi
        topology = TileTopology(
            #TODO clear tmp folder after use ?
            image=wsi, tile_builder=CytomineTileBuilder('tmp'),
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
            self._kept_tile_map = np.where([not skip_tile(tile.identifier,
                                         topology) for tile in topology])[0] + 1

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
