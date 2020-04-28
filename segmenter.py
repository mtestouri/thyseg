import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import cv2
from glob import glob
from shutil import copyfile
from dataset import ImgSet

class Segmenter:
    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.model = self.init_model().to(self.device)

    def init_model(self):
        raise NotImplementedError

    def save_model(self, model_file):
        torch.save(self.model.state_dict(), model_file)

    def load_model(self, model_file):
        self.model.load_state_dict(torch.load(model_file))

    def train(self, dataset, n_epochs):
        raise NotImplementedError

    def predict(self, images):
        self.model.eval() #TODO not a problem here ? -> see integration
        with torch.no_grad():
            return torch.sigmoid(self.model(images.to(self.device)))

    def segment(self, dataset, dest='segmentations', batch_size=1, psize=None, 
                norm=False, blur_ks=None, thresh=None, assess=False):
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
        # compute segmentations
        count = 0
        dl = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=2)
        for i, (images, masks, files_id) in enumerate(dl):
            im_h = images[0].shape[1]
            im_w = images[0].shape[2]
            if not psize_given:
                patch_h = im_h
                patch_w = im_w
            if (im_h % patch_h) != 0 or (im_w % patch_w) != 0:
                raise ValueError("the patch size must divide the image dimensions")
            # compute masks using a sliding patch
            masks_p = np.zeros((batch_size, im_h, im_w, 3), dtype=np.float32)
            i_h = 0
            for j in range(im_h // patch_h):
                i_w = 0
                for k in range(im_w // patch_w):
                    patchs = images[:, :, i_h:(i_h+patch_h), i_w:(i_w+patch_w)]
                    # predict masks
                    preds = self.predict(patchs)
                    preds = preds.permute(0, 2, 3, 1).cpu().numpy()
                    # recreate mask patchs from classes
                    m_h = preds[0].shape[0]
                    m_w = preds[0].shape[1]
                    for l in range(len(preds)):
                        mask = preds[l, :, :, 1].reshape(m_h, m_w, 1)
                        # post-processing
                        if norm:
                            mask = (mask - np.mean(mask))/np.std(mask)
                        if blur_ks is not None:
                            if blur_ks < 0:
                                raise ValueError("blur size must be greater than 0")
                            mask = cv2.blur(mask, (blur_ks, blur_ks))
                        if thresh is not None:
                            if thresh < 0 or thresh > 1:
                                raise ValueError("threshold must belong to [0,1]")
                            _, mask = cv2.threshold(mask, thresh, 1, cv2.THRESH_BINARY)
                        mask = mask.reshape(m_h, m_w, 1)
                        # convert to RGB image
                        if assess:
                            z = np.zeros((m_h, m_w, 1), dtype=np.float32)
                            mask = np.concatenate((z, mask, z), axis=2)*180
                        else:
                            mask = np.concatenate((mask, mask, mask), axis=2)*255
                        # resize if necessary
                        if mask.shape != (patch_h, patch_w):
                            mask = cv2.resize(mask, (patch_w, patch_h))
                        # write mask patch
                        masks_p[l, i_h:(i_h+patch_h), i_w:(i_w+patch_w)] = mask

                    i_w += patch_w
                i_h += patch_h
            # write the image files
            for j in range(len(images)):
                image = images[j]
                mask = masks[j]
                mask_p = masks_p[j]

                if assess:
                    z = np.zeros((im_h, im_w, 1), dtype=np.float32)
                    sep = np.ones((im_h, 10, 3), dtype=np.float32)*255
                    # convert tensors to numpy
                    image = image.permute(1, 2, 0).cpu().numpy()
                    mask = mask.permute(1, 2, 0).numpy()
                    # recreate image mask from classes
                    mask = mask[:, :, 1].reshape(im_h, im_w, 1)*180
                    mask = np.concatenate((z, mask, z), axis=2)
                    # combine masks and images
                    sup = image + mask
                    sup_p = image + mask_p
                    # add labels
                    cv2.putText(mask, "man", (10, 30), fontScale=1, thickness=2, 
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 255))
                    cv2.putText(mask_p, "pred", (10, 30), fontScale=1, thickness=2,
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 255))
                    # final image
                    img = np.concatenate((image, sep, 
                                          sup, sep, 
                                          sup_p, sep, 
                                          mask, sep, 
                                          mask_p), axis=1)
                    cv2.imwrite(dest + "/seg" + str(count) + ".jpg", img)
                else: # just the mask
                    cv2.imwrite(dest  + "/" + files_id[j] + "_y.jpg", mask_p)
                count += 1
                print(f'segmentation: {count}/{len(dataset)}', end='\r')
        print("\nsegmentation done")

    def iter_data_imp(self, folder, n_iters, n_epochs, blur_ks=5, thresh=0.5):
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
            self.segment(ImgSet(dest), dest=dest, norm=True, blur_ks=blur_ks,
                         thresh=thresh)