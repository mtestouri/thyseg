import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import cv2

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

    def train(self, dataset, num_epochs):
        raise NotImplementedError

    def segment(self, dataset, patch_size=None):
        print("segmenting..")
        # create folder
        if not os.path.exists('segmentations'):
            os.makedirs('segmentations')
        # used to rebuild the images
        (x, _) = dataset[0]
        im_h = x.shape[1]
        im_w = x.shape[2]
        if patch_size is None:
            patch_size = im_h
        if patch_size < 1:
            raise ValueError("the patch size must be greater than 0")
        if (im_h % patch_size) != 0 or (im_w % patch_size) != 0:
            raise ValueError("the patch size must divide the image dimensions")
        z = np.zeros((im_h, im_w, 1), dtype=np.uint8)
        sep = np.ones((im_h, 10, 3), dtype=np.uint8)*255
        # compute segmentations
        data_loader = DataLoader(dataset=dataset, batch_size=1, num_workers=2)
        for i, (x, y) in enumerate(data_loader):
            y_pred = np.zeros((im_h, im_w, 2), dtype=np.float32)
            i_h = 0
            for j in range(x.shape[2] // patch_size):
                i_w = 0
                for k in range(x.shape[3] // patch_size):
                    # extract patch
                    patch = x[:, :, i_h:(i_h+patch_size), i_w:(i_w+patch_size)]
                    # compute mask
                    with torch.no_grad():
                        patch = patch.to(self.device)
                        p_y_pred = torch.sigmoid(self.model(patch))
                        p_y_pred = p_y_pred.permute(0, 2, 3, 1).squeeze(0).cpu().numpy()
                        # debug
                        #import matplotlib.pyplot as plt
                        #plt.imshow(patch.permute(0, 2, 3, 1).cpu().squeeze(0).int())
                        #plt.show()
                        #c = np.int32(p_y_pred[:, :, 1].reshape(128, 128, 1)*180)
                        #a = np.zeros((128, 128, 1), dtype=np.uint8)
                        #c = np.concatenate((a, c, a), axis=2)
                        #plt.imshow(c)
                        #plt.show()
                    # write patch
                    y_pred[i_h:(i_h+patch_size), i_w:(i_w+patch_size)] = p_y_pred
                    i_w += patch_size
                i_h += patch_size
            # convert tensors to numpy
            x = x.permute(0, 2, 3, 1).squeeze(0).cpu().numpy()
            y = y.permute(0, 2, 3, 1).squeeze(0).numpy()
            # recreate image masks from classes
            y = np.concatenate(
                (z, y[:, :, 1].reshape(im_h, im_w, 1)*180, z), axis=2)
            y_pred = np.concatenate(
                (z, y_pred[:, :, 1].reshape(im_h, im_w, 1)*180, z), axis=2)
            # combine masks and images
            sup_y = x + y
            sup_y_pred = x + y_pred
            # add labels
            cv2.putText(y, "man", (10, 30), fontScale=1, thickness=2, 
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 255))
            cv2.putText(y_pred, "pred", (10, 30), fontScale=1, thickness=2,
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 255))
            # write files
            img = np.concatenate((x, sep,
                                 sup_y, sep,
                                 sup_y_pred, sep,
                                 y, sep,
                                 y_pred), axis=1)
            cv2.imwrite("segmentations/seg" + str(i) + ".jpg", img)
            print(f'segmentation: {i+1}/{len(dataset)}', end='\r')
        print("\nsegmentation done")