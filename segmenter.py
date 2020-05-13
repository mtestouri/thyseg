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

# TODO size 'man' et 'pred'
# TODO union instead of replace in idi

class Segmenter:
    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.model = None

    def check_model_init(self):
        if self.model is None:
            raise ValueError("variable 'self.model' not initialized")

    def save_model(self, model_file):
        self.check_model_init()
        torch.save(self.model.state_dict(), model_file)

    def load_model(self, model_file):
        self.check_model_init()
        self.model.load_state_dict(torch.load(model_file))

    def train(self, dataset, n_epochs):
        raise NotImplementedError

    def predict(self, images, transform=None):
        self.check_model_init()
        self.model.eval() #TODO not a problem here ? -> see integration
        # compute masks
        with torch.no_grad():
            masks = torch.sigmoid(self.model(images.to(self.device)))
        # post-processing
        if transform is not None:
            for i in range(len(masks)):
                masks[i] = transform(masks[i].cpu()).to(self.device)
        return masks

    def segment(self, dataset, dest='segmentations', batch_size=1, psize=None, 
                transform=None, assess=False):
        print("segmenting..")
        psize_given = (psize is not None)
        if psize_given:
            if psize < 1:
                raise ValueError("the patch size must be greater than 0")
            else:
                patch_h = psize
                patch_w = psize
        # create folder
        while dest[-1] == '/':
            dest = dest[:len(dest)-1]
        if not os.path.exists(dest):
            os.makedirs(dest)
        
        if assess:
            sum_dice = 0
            sum_jaccard = 0
        tf_resize = Resize()
        dl = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=2)
        for i, (images, masks, files_id) in enumerate(dl):
            # check dimensions
            im_h = images[0].shape[1]
            im_w = images[0].shape[2]
            if not psize_given:
                patch_h = im_h
                patch_w = im_w
            if (im_h % patch_h) != 0 or (im_w % patch_w) != 0:
                raise ValueError("the patch size must divide the image dimensions")
            
            # compute masks using a sliding patch
            masks_p = torch.zeros((batch_size, 2, im_h, im_w), dtype=torch.float32)
            i_h = 0
            for j in range(im_h // patch_h):
                i_w = 0
                for k in range(im_w // patch_w):
                    patchs = images[:, :, i_h:(i_h+patch_h), i_w:(i_w+patch_w)]
                    preds = self.predict(patchs)
                    
                    for l in range(len(preds)):
                        mask = preds[l].cpu()
                        # resize if necessary
                        if mask.shape != (2, patch_h, patch_w):
                            mask = tf_resize(mask, (patch_w, patch_h))
                        # write mask patch
                        masks_p[l, :, i_h:(i_h+patch_h), i_w:(i_w+patch_w)] = mask
                    i_w += patch_w
                i_h += patch_h
            
            for j in range(len(images)):
                image = images[j]
                mask = masks[j]
                mask_p = masks_p[j]
                # post-processing
                if transform is not None:
                    mask_p = transform(mask_p)

                if assess:
                    # metrics
                    sum_dice += dice(mask_p, mask).item()
                    sum_jaccard = jaccard(mask_p, mask).item()
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
                    # add labels
                    cv2.putText(mask, "man", (10, 30), fontScale=1, thickness=2, 
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 255))
                    cv2.putText(mask_p, "pred", (10, 30), fontScale=1, thickness=2,
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 255))
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