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

from cytomine.models import ImageInstance, Annotation, AnnotationCollection
from shapely.affinity import affine_transform
from sldc import TileTopology, SemanticLocator, SemanticMerger
from sldc_cytomine import CytomineSlide, CytomineTileBuilder
from sldc_openslide import OpenSlideImage, OpenSlideTileBuilder


FOREGROUND = 154005477  # foreground term id
SIMPLIFY_TOLERANCE = 4  # tolerance for polygons simplification
CACHE = "__cache__"     # path were the images are cached


class Segmenter:
    """
    generic segmenter

    parameters
    ----------
    c_weights: float array
        class weights used for loss computation
    """
    
    def __init__(self, c_weights=None):
        if torch.cuda.is_available():
            self._device = torch.device('cuda')
        else:
            self._device = torch.device('cpu')
        self._model = None
        self._c_weights = c_weights

    def _check_model_init(self):
        if self._model is None:
            raise ValueError("variable 'self._model' not initialized")

    def save_model(self, model_file):
        """
        save the model weights

        parameters
        ----------
        model_file: string
            file where the model is saved
        """

        self._check_model_init()
        torch.save(self._model.state_dict(), model_file)

    def load_model(self, model_file):
        """
        load the model weights

        parameters
        ----------
        model_file: string
            model file to load\n
            beware that the depth of the segmenter and the model file must match
        """

        self._check_model_init()
        self._model.load_state_dict(torch.load(model_file))

    def set_eval(self):
        """
        set the model in evaluation mode
        """
        self._model.eval()

    def set_train(self):
        """
        set the model in training mode
        """
        self._model.train()

    def train(self, dataset, n_epochs):
        raise NotImplementedError

    def predict(self, images, transform=None):
        """
        segment a batch of images

        parameters
        ----------
        images: tensor
            images tensor of shape: (batch_size, n_channels, height, width)

        transform: Transform
            transform to apply to the predicted masks

        returns
        -------
        masks: tensor
            masks tensor of shape: (batch_size, n_channels, height, width)
        """

        self._check_model_init()
        # check eval mode
        train_mode = self._model.training
        if train_mode:
            self._model.eval()
        
        with torch.no_grad():
            # compute masks
            masks = torch.sigmoid(self._model(images.to(self._device)))
        
            # post-processing
            if transform is not None:
                for i in range(len(masks)):
                    masks[i] = transform(masks[i].cpu()).to(self._device)
            
        # optionally restore train mode
        if train_mode:
            self._model.train()
        
        return masks

    def _segment_core(self, images, tsize=None, transform=None):
        """
        segment a batch of images using tiles

        parameters
        ----------
        images: tensor
            images tensor of shape: (batch_size, n_channels, height, width)

        tsize: int
            tile size

        transform: Transform
            transform to apply to the predicted masks

        returns
        -------
        masks: tensor
            masks tensor of shape: (batch_size, n_channels, height, width)
        """

        # image dimensions
        im_h = images[0].shape[1]
        im_w = images[0].shape[2]

        # init tsize
        if tsize is not None:
            if tsize < 1:
                raise ValueError("'tsize' must be greater than 0")
            else:
                tile_h = tsize
                tile_w = tsize

            # check dimensions
            if (im_h % tile_h) != 0 or (im_w % tile_w) != 0:
                raise ValueError("the tile size must divide the image dimensions") 
        else:
            tile_h = im_h
            tile_w = im_w
        
        # inits
        self.set_eval()
        tf_resize = Resize()
        masks = torch.zeros((images.shape[0], 2, im_h, im_w), dtype=torch.float32)
        
        # compute masks using tiles
        i_h = 0
        for i in range(im_h // tile_h):
            i_w = 0
            for j in range(im_w // tile_w):
                # extract tiles
                tiles = images[:, :, i_h:(i_h+tile_h), i_w:(i_w+tile_w)]
                # compute tile masks
                preds = self.predict(tiles)
                    
                for k in range(preds.shape[0]):
                    mask = preds[k].cpu()
                    # resize if necessary
                    if mask.shape != (2, tile_h, tile_w):
                        mask = tf_resize(mask, (tile_w, tile_h))
                    
                    # write tile mask
                    masks[k, :, i_h:(i_h+tile_h), i_w:(i_w+tile_w)] = mask

                i_w += tile_w
            i_h += tile_h

        # post-processing
        if transform is not None:
            for i in range(masks.shape[0]):
                masks[i] = transform(masks[i])

        return masks

    def segment_folder(self, folder, dest='segmentations', tsize=None, 
                       batch_size=1, transform=None, assess=False):
        """
        segment a folder

        parameters
        ----------
        folder: string
            path to folder to be segmented

        dest: string
            predicted masks destination folder

        tsize: int
            tile size

        batch_size: int
            batch size

        transform: Transform
            transform to apply to the predicted masks

        assess: bool
            toggle assessment mode which will compute the metrics and write the 
            predicted masks along with their ground truth counterparts
        """

        print("segmenting..")
        # create dataset
        dataset = ImgSet(folder, assess)
        dl = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=2)
        
        # image dimensions
        im_h = dataset.im_h
        im_w = dataset.im_w

        # create masks destination folder
        while dest[-1] == '/':
            dest = dest[:len(dest)-1]
        if not os.path.exists(dest):
            os.makedirs(dest)
        
        if assess:
            sum_dice = 0
            sum_jaccard = 0        
            
            for i, (images, masks, files_id) in enumerate(dl):
                # compute predicted masks
                masks_p = self._segment_core(images, tsize, transform)
            
                for j in range(images.shape[0]):
                    image = images[j]
                    mask_p = masks_p[j]
                    mask = masks[j]

                    # metrics
                    mask = mask.unsqueeze(0)
                    mask_p = mask_p.unsqueeze(0)
                    sum_dice += dice(mask_p, mask, self._c_weights).item()
                    sum_jaccard += jaccard(mask_p, mask, self._c_weights).item()
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

                    # display info
                    print("segmentation: " + str(i+1) + "/" + str(len(dataset))
                          + ", avg_dice: " + str(round(sum_dice/(i+1), 4))
                          + ", avg_jaccard: " + str(round(sum_jaccard/(i+1), 4))
                          , end='\r')

        else:
            for i, (images, files_id) in enumerate(dl):
                # compute predicted masks
                masks = self._segment_core(images, tsize, transform)

                for j in range(masks.shape[0]):
                    mask = masks[j]

                    # convert tensor to numpy
                    mask = mask.permute(1, 2, 0).numpy()
                    
                    # select foreground channel
                    mask = mask[:, :, 1].reshape(im_h, im_w, 1)*255
                    
                    # convert to RGB image
                    mask = np.concatenate((mask, mask, mask), axis=2)
                    
                    # write final image
                    cv2.imwrite(dest  + "/" + files_id[j] + "_y.jpg", mask)
    
                    # display info
                    print(f'segmentation: {(i+1)}/{len(dataset)}', end='\r')
        
        print("\nsegmentation done")

    def iter_data_imp(self, folder, n_iters, n_epochs, transform=None):
        """
        iterative data improvement of a dataset

        parameters
        ----------
        folder: string
            folder of the dataset to improve

        n_iters: int
            number of improvement iterations

        n_epochs: int
            number of epochs for training

        transform: Transform
            transform to apply to the predicted masks
        """

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
            self.train(dest, n_epochs)
            self.segment_folder(dest, dest=dest, transform=transform)
            
            # merge ground truth masks with predicted masks
            for y_file in y_files:
                y = cv2.imread(y_file)
                y_pred = cv2.imread(dest + "/" + y_file[len(folder)+1:])

                if y.shape != y_pred.shape:
                    y_pred = cv2.resize(y_pred, (y.shape[1], y.shape[0]),
                                        cv2.INTER_LINEAR)
                
                y_pred = cv2.bitwise_or(y, y_pred)
                cv2.imwrite(dest  + "/" + y_file[len(folder)+1:], y_pred)

    def segment_wsi(self, image_id, window=[], tsize=512, batch_size=4, 
                    transform=None, slide_type='openslide'):
        """
        remote segmentation of a WSI

        parameters
        ----------
        image_id: int
            id of the WSI to segment

        windows: int array
            the window where to perform the segmentation, the window is in  
            the form : [off_x, off_y, width, height] and the origin is the lower 
            left corner

        tsize: int
            tile size

        batch_size: int
            batch size

        transform: Transform
            transform to apply to the predicted masks

        slide_type: string
            slide type : 'openslide' or 'cytomineslide'
        """

        # fetch wsi instance
        image_instance = ImageInstance().fetch(image_id)

        # window parameters
        if len(window) == 4:
            off_x = window[0]
            off_y = window[1]
            w_width = window[2]
            w_height = window[3]
            complete = False
        else:
            off_x = 0
            off_y = 0
            w_width = image_instance.width
            w_height = image_instance.height
            complete = True

        # overlap between tiles
        overlap = int(round(tsize / 4))

        # create OpenSlide image
        if slide_type == 'openslide':
            # image file
            img_file = (CACHE + "/"
                        + str(image_instance.id) + "-"
                        + str(off_x) + "-"
                        + str(image_instance.height - off_y - w_height) + "-"
                        + str(w_width) + "-"
                        + str(w_height)
                        + ".jpg")
            
            # create cache folder
            if not os.path.exists(CACHE):
                os.makedirs(CACHE)

            # download image file if necessary
            if not os.path.isfile(img_file):
                if complete:
                    image_instance.download(dest_pattern=img_file)
                else:
                    image_instance.window(off_x, image_instance.height - off_y
                                          - w_height, w_width, w_height, 
                                          dest_pattern=img_file)
        
            # create slide image
            wsi = OpenSlideImage(img_file)
            if (wsi.width != w_width) or (wsi.height != w_height):
                raise ValueError("invalid image dimensions: expected " 
                                 + str((w_width, w_height)) + ", got "
                                 + str((wsi.width, wsi.height)))
            # create dataset
            dataset = TileDataset(wsi, tsize, tsize, overlap, 'openslide')

        # create CytomineSlide image
        elif slide_type == 'cytomineslide':
            # create slide image
            if complete:
                wsi = CytomineSlide(image_id)
            else:
                wsi = CytomineSlide(image_id).window((off_x, 
                    image_instance.height - off_y - w_height), w_width, w_height)
            # create dataset
            dataset = TileDataset(wsi, tsize, tsize, overlap, 'cytomineslide')
            
        else:
            raise ValueError("invalid slide type: " + str(slide_type))

        # inits
        count = 0
        self.set_eval()
        locator = SemanticLocator(background=0)
        tile_polygons, tile_ids = list(), list()

        # compute mask polygons
        dl = DataLoader(dataset=dataset, batch_size=batch_size)
        for x, ids in dl:
            # convert to from RGB to BGR tensors
            x = x[:, :, :, [2, 1, 0]].permute(0, 3, 1, 2).float()                
            # predict tile masks
            y = self.predict(x, transform)
            # select foreground channel
            y = y.permute(0, 2, 3, 1)[:, :, :, 1].cpu().numpy()

            # turn prediction into polygons
            for i in range(y.shape[0]):
                polygons = locator.locate(y[i], 
                                    offset=dataset.topology.tile_offset(ids[i]))
                if len(polygons) > 0:
                    polygons, _ = zip(*polygons)
                    polygons = list(polygons)

                    # simplify the polygons to reduce memory usage
                    for j in range(len(polygons)):
                        polygons[j] = polygons[j].simplify(SIMPLIFY_TOLERANCE)

                    tile_polygons.append(polygons)

                else:
                    tile_polygons.append(list())

            tile_ids.extend(ids.numpy())
            count += x.shape[0]
            print(f'processed tiles {count}/{len(dataset)}')
        
        # merge polygons overlapping several tiles
        print("merging polygons..")
        merged = SemanticMerger(tolerance=1).merge(tile_ids, tile_polygons,
                                                   dataset.topology)
        
        # upload to cytomine
        print("uploading annotations..")
        anns = AnnotationCollection()
        for polygon in merged:
            anns.append(
                Annotation(
                location=self._change_referential(polygon, 
                                                  off_x, off_y, w_height).wkt,
                id_image=image_instance.id,
                id_project=image_instance.project,
                term=[FOREGROUND]
                )
            )
        anns.save(n_workers=4)

    def _change_referential(self, p, off_x, off_y, w_height):
        return affine_transform(p, [1, 0, 0, -1, off_x, off_y + w_height])

    def _skip_tile(self, tile_id, topology):
        tile_row, tile_col = topology._tile_coord(tile_id)
        skip_bottom = (topology._image.height 
                       % (topology._max_height - topology._overlap)) != 0
        skip_right = (topology._image.width 
                      % (topology._max_width - topology._overlap)) != 0
        return (skip_bottom and tile_row == topology.tile_vertical_count - 1) or \
               (skip_right and tile_col == topology.tile_horizontal_count - 1)


class ImgSet(Dataset):
    """
    image dataset

    parameters
    ----------
    folder: string
        folder of the dataset to load, files are in the form 
        'id_x.jpg' (images) and 'id_y.jpg' (masks)

    masks: bool
        toggle the loading of the masks
    """

    def __init__(self, folder, masks=True):
        while folder[-1] == '/':
            folder = folder[:len(folder)-1]
        self._df = folder
        self._masks = masks
        
        # load list of files
        self._files = glob(self._df + "/*_x.jpg")
        if len(self._files) == 0:
            raise FileNotFoundError("no files found in folder '" + self._df + "'")
        
        # define dataset image size
        self._im_h = None
        self._im_w = None
        if self._masks:
            (x, _, _) = self.__getitem__(0)
        else:
            (x, _) = self.__getitem__(0)
        self._im_h = x.shape[1]
        self._im_w = x.shape[2]

    @property
    def im_h(self):
        return self._im_h

    @property
    def im_w(self):
        return self._im_w
    
    def __getitem__(self, index):
        # load image
        x_file = self._files[index]
        file_id = x_file[len(self._df)+1:len(x_file)-6]
        x = cv2.imread(x_file)
        
        # load mask
        if self._masks:
            y_file = self._df + "/" + file_id + "_y.jpg"
            y = cv2.imread(y_file)
            if y is None:
                raise FileNotFoundError("unable to load '" + y_file + "'")
        
        # check size
        if (self._im_h is not None) and (self._im_w is not None):
            if (x.shape[0] != self._im_h) or (x.shape[1] != self._im_w):
                x = cv2.resize(x, (self._im_w, self._im_h), cv2.INTER_LINEAR)
            if self._masks:
                if (y.shape[0] != self._im_h) or (y.shape[1] != self._im_w):
                    y = cv2.resize(y, (self._im_w, self._im_h), cv2.INTER_LINEAR) 
        
        # convert to tensor
        x = torch.from_numpy(x).float().permute(2, 0, 1)
        
        if self._masks:
            # RGB masks to classe masks
            y = np.abs(np.round(y/255)[:, :, :2] - (1, 0))
            # convert to tensor
            y = torch.from_numpy(y).float().permute(2, 0, 1)
            return x, y, file_id
        else:
            return x, file_id

    def __len__(self):
        return len(self._files)


class TileDataset(Dataset):
    """
    SLDC tile dataset

    parameters
    ----------
    wsi: Image
        OpenSlideImage or CytomineSlide object
        
    tile_width: int
        tile width

    tile_height: int
        tile height

    overlap: int
        overlap between tiles

    slide_type: string
        slide type : 'openslide' or 'cytomineslide'
    """
    
    def __init__(self, wsi, tile_width, tile_height, overlap, slide_type):
        # tile builder
        if slide_type == 'openslide':
            tile_builder =  OpenSlideTileBuilder()
        elif slide_type == 'cytomineslide':
            tile_builder = CytomineTileBuilder(CACHE)
        else:
            raise ValueError("invalid slide type: " + str(slide_type))
        
        self._wsi = wsi
        self._topology = TileTopology(
            image=wsi, tile_builder=tile_builder,
            max_width=tile_width, max_height=tile_height,
            overlap=overlap
        )
        self._tile_width = tile_width
        self._tile_height = tile_height
        self._overlap = overlap

    @property
    def topology(self):
        return self._topology

    def __getitem__(self, index):
        """
        returns (numpy image of a tile, tile identifier)
        """
        identifier = index + 1
        tile_np = self._topology.tile(identifier).np_image
        
        # right padding
        if tile_np.shape[0] != self._tile_width:
            pad = np.zeros((self._tile_width - tile_np.shape[0], 
                            tile_np.shape[1], tile_np.shape[2]), dtype=np.uint8)
            tile_np = np.concatenate((tile_np, pad), axis=0)

        # bottom padding
        if tile_np.shape[1] != self._tile_height:
            pad = np.zeros((tile_np.shape[0], self._tile_height - tile_np.shape[1], 
                            tile_np.shape[2]), dtype=np.uint8)
            tile_np = np.concatenate((tile_np, pad), axis=1)

        return tile_np, identifier

    def __len__(self):
        return self.topology.tile_count
