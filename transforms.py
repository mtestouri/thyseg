import torch, cv2
import torch.nn.functional as F
import numpy as np


class Threshold:
    def __init__(self, thresh=0.5):
        if thresh < 0 or thresh > 1:
            raise ValueError("'thresh' must belong to [0,1]")
        self._thresh = thresh

    def __call__(self, images):
        """
        apply a threshold to a batch of images

        parameters
        ----------
        images: tensor
            batch of images of shape (batch_size, n_channels, height, width)

        returns
        -------
        images: tensor
            batch of images of shape (batch_size, n_channels, height, width)
        """
        #TODO port to torch

        for i in range(len(images)):
            image = images[i].permute(1, 2, 0).numpy()
            _, image = cv2.threshold(image, self._thresh, 1, cv2.THRESH_BINARY)
            if len(image.shape) < 3:
                image = np.expand_dims(image, 2)
            images[i] = torch.from_numpy(image).permute(2, 0, 1)
        return images


class Resize:
    def __call__(self, images, size):
        """
        resize a batch of images

        parameters
        ----------
        images: tensor
            batch of images of shape (batch_size, n_channels, height, width)
        
        size: tuple
            new size: (n_height, n_width)

        returns
        -------
        images: tensor
            batch of images of shape (batch_size, n_channels, n_height, n_width)
        """

        return F.interpolate(images, size, mode="bilinear")