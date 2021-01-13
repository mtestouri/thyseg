import os
import torch, cv2
from torch.utils.data import DataLoader
from transforms import Resize
from datasets import ImgDataset, TileDataset
from metrics import jaccard


class Segmenter:
    """
    generic segmenter

    parameters
    ----------
    n_classes: int
        number of classes

    c_weights: float array
        class weights used for loss computation

    device: string
        device to use for segmentation : 'cpu' or 'cuda'
    """
    
    def __init__(self, n_classes=2, c_weights=None, device='cuda'):
        self._model = None
        
        # check n_classes
        if n_classes < 1:
            raise ValueError("'n_classes' must be greater or equal to 1")
        self._n_classes = n_classes
        
        # check c_weights
        self._c_weights = None
        if c_weights is not None:
            if (len(c_weights) != n_classes):
                raise ValueError("number of weights must be equal to the number of classes")
            self._c_weights = torch.Tensor(c_weights)

        # check device
        self._check_device(device)
        if device == 'cuda':
            if torch.cuda.is_available():
                self._device = torch.device('cuda')
            else:
                raise Exception("CUDA is not available")
        else:
            self._device = torch.device('cpu')

    def _check_model_init(self):
        if self._model is None:
            raise ValueError("variable 'self._model' not initialized")

    def _check_device(self, device):
        if device != 'cuda' and device != 'cpu':
            raise ValueError("invalid device type: " + str(device))

    def set_device(self, device):
        """
        assign the segmenter to a device and move the model to it

        parameters
        ----------
        device: string
            device to use for segmentation : 'cpu' or 'cuda'
        """
        
        # checks
        self._check_model_init()
        self._check_device(device)
        # move to new device
        self._device = torch.device(device)
        self._model = self._model.to(self._device)

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
            model file to load
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

    def train(self, folder, n_epochs, tsize=512):
        raise NotImplementedError

    def predict(self, images, transform=None):
        """
        segment a batch of images

        parameters
        ----------
        images: tensor
            images tensor of shape: (batch_size, n_channels, height, width)

        transform: Transform
            transform applied to the predicted masks

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
            masks = self._model(images.to(self._device))
            # post-processing
            if transform is not None:
                masks = transform(masks.cpu()).to(self._device)
            
        # optionally restore train mode
        if train_mode:
            self._model.train()
        
        return masks

    def segment(self, folder, dest="", tsize=512, transform=None):
        """
        segment a folder of images

        parameters
        ----------
        folder: string
            folder containing the images to segment

        dest: string
            folder where the predicted masks are written

        tsize: int
            segmentation tile size

        transform: Transform
            transform applied to the predicted masks
        """
        
        # inits
        self.set_eval()
        tf_resize = Resize()
        dataset = ImgDataset(folder)
        img_count = 0
        mask_p = None
        sum_jaccard = 0

        for image, mask in dataset:
            img_count += 1
            sum_intersion, sum_union = 0, 0
            tile_dataset = TileDataset(image, mask, tsize=tsize,
                                       mask_merge=(self._n_classes <= 2))
            dl = DataLoader(dataset=tile_dataset, batch_size=1)

            for tile, tile_mask, tile_id in dl:
                # compute tile position and size without padding
                offset = tile_dataset.topology.tile_offset(tile_id)
                off_x, off_y = offset[1].item(), offset[0].item()
                t_h, t_w = tsize, tsize
                if off_x + tsize > image.height:
                    t_h = image.height - off_x
                if off_y + tsize > image.width:
                    t_w = image.width - off_y

                # compute predicted tile mask
                tile_mask_p = self.predict(tile, transform).cpu()
                # resize if necessary
                if tile_mask_p.shape != tile_mask.shape:
                    tile_mask_p = tf_resize(tile_mask_p, (tsize, tsize))
                # select area without padding
                tile_mask_p = tile_mask_p[:,1:,:t_h,:t_w].int()
                tile_mask = tile_mask[:,1:,:t_h,:t_w].int()

                # compute intersection and union
                sum_intersion += torch.sum(torch.bitwise_and(tile_mask_p, tile_mask))
                sum_union += torch.sum(torch.bitwise_or(tile_mask_p, tile_mask))
                #TODO tile overlap and merging should be taken into account when computing IoU
                #TODO this is computed on the whole tensor -> channel wise better ?
                
                # write the predicted tile mask to the predicted mask
                # bitwise_or is used for tiles merging
                if mask_p is None:
                    mask_p = torch.zeros(tile_mask_p.shape[1], image.height, 
                                         image.width, dtype=torch.int)
                mask_p[:, off_x:(off_x+t_h), off_y:(off_y+t_w)] = torch.bitwise_or(
                    mask_p[:, off_x:(off_x+t_h), off_y:(off_y+t_w)], tile_mask_p.squeeze(0))

            # write the predicted mask channels to files
            if dest == "":
                dest = folder
            if not os.path.exists(dest):
                os.makedirs(dest)
            for i in range(mask_p.shape[0]):
                filename = dest + f'/{img_count}_yp_{i+1}.png'
                cv2.imwrite(filename, mask_p[i].numpy()*255)

            # compute jaccard
            jaccard = 1
            if sum_union != 0:
                jaccard = sum_intersion / sum_union
            sum_jaccard += jaccard
            print(f'image: {img_count}/{len(dataset)}, jaccard: {jaccard:.4f}')
        print(f'average jaccard: {(sum_jaccard/len(dataset)):.4f}')