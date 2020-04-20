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

    def predict(self, image):
        with torch.no_grad():
            return torch.sigmoid(self.model(image.to(self.device)))

    def segment(self, dataset, psize=None, assess=False):
        print("segmenting..")
        psize_given = (psize is not None)
        if psize_given:
            if psize < 1:
                raise ValueError("the patch size must be greater than 0")
            else:
                psize_h = psize
                psize_w = psize
        # create folder
        if not os.path.exists('segmentations'):
            os.makedirs('segmentations')
        # compute segmentations
        data_loader = DataLoader(dataset=dataset, batch_size=1, num_workers=2)
        for i, (x, y) in enumerate(data_loader):
            im_h = x.shape[2]
            im_w = x.shape[3]
            if not psize_given:
                psize_h = im_h
                psize_w = im_w
            if (im_h % psize_h) != 0 or (im_w % psize_w) != 0:
                raise ValueError("the patch size must divide the image dimensions")
            # compute mask using a sliding patch
            y_pred = np.zeros((im_h, im_w, 3), dtype=np.float32)
            i_h = 0
            for j in range(x.shape[2] // psize_h):
                i_w = 0
                for k in range(x.shape[3] // psize_w):
                    # extract patch
                    patch = x[:, :, i_h:(i_h+psize_h), i_w:(i_w+psize_w)]
                    # predict mask
                    mask = self.predict(patch)
                    mask = mask.permute(0, 2, 3, 1).squeeze(0).cpu().numpy()
                    # recreate patch mask from classes
                    m_h = mask.shape[0]
                    m_w = mask.shape[1]
                    mask = mask[:, :, 1].reshape(m_h, m_w, 1)
                    if assess:
                        z = np.zeros((m_h, m_w, 1), dtype=np.float32)
                        mask = np.concatenate((z, mask, z), axis=2)*180
                    else:
                        mask = np.concatenate((mask, mask, mask), axis=2)*255
                    # resize if necessary
                    if mask.shape != (psize_h, psize_w):
                        mask = cv2.resize(mask, (psize_w, psize_h))
                    # write patch
                    y_pred[i_h:(i_h+psize_h), i_w:(i_w+psize_w)] = mask
                    i_w += psize_w
                i_h += psize_h
            # write the image file
            if assess:
                z = np.zeros((im_h, im_w, 1), dtype=np.float32)
                sep = np.ones((im_h, 10, 3), dtype=np.float32)*255
                # convert tensors to numpy
                x = x.permute(0, 2, 3, 1).squeeze(0).cpu().numpy()
                y = y.permute(0, 2, 3, 1).squeeze(0).numpy()
                # recreate image mask from classes
                y = y[:, :, 1].reshape(im_h, im_w, 1)*180
                y = np.concatenate((z, y, z), axis=2)
                # combine masks and images
                sup_y = x + y
                sup_y_pred = x + y_pred
                # add labels
                cv2.putText(y, "man", (10, 30), fontScale=1, thickness=2, 
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 255))
                cv2.putText(y_pred, "pred", (10, 30), fontScale=1, thickness=2,
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 255))
                # final image
                img = np.concatenate((x, sep, 
                                      sup_y, sep, 
                                      sup_y_pred, sep, 
                                      y, sep, 
                                      y_pred), axis=1)
            else:
                img = y_pred # just the mask
            cv2.imwrite("segmentations/seg" + str(i) + ".jpg", img)
            print(f'segmentation: {i+1}/{len(dataset)}', end='\r')
        print("\nsegmentation done")