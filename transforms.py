import torch
import cv2
import numpy as np

class Normalize:
    def __call__(self, image):
        for i in range(len(image)):
            image[i] = (image[i] - torch.mean(image[i]))/torch.std(image[i])
        return image

class Smoothing:
    def __init__(self, ksize=10):
        if ksize < 1:
            raise ValueError("'ksize' must be greater or equal to 1")
        self.ksize = ksize

    def __call__(self, image):
        image = image.permute(1, 2, 0).numpy()
        image = cv2.blur(image, (self.ksize, self.ksize))
        return torch.from_numpy(image).permute(2, 0, 1)

class ErodeDilate:
    def __init__(self, n_iters=2, ksize=10):
        if n_iters < 1:
            raise ValueError("'n_iters' must be greater or equal to 1")
        if ksize < 1:
            raise ValueError("'ksize' must be greater or equal to 1")
        self.n_iters = n_iters
        self.kernel = np.ones((ksize, ksize), np.uint8)
    
    def __call__(self, image):
        image = image.permute(1, 2, 0).numpy()
        image = cv2.erode(image, self.kernel, iterations=self.n_iters)
        image = cv2.dilate(image, self.kernel, iterations=self.n_iters)
        return torch.from_numpy(image).permute(2, 0, 1)

class Threshold:
    def __init__(self, thresh=0.5):
        if thresh < 0 or thresh > 1:
            raise ValueError("'thresh' must belong to [0,1]")
        self.thresh = thresh

    def __call__(self, image):
        image = image.permute(1, 2, 0).numpy()
        _, image = cv2.threshold(image, self.thresh, 1, cv2.THRESH_BINARY)
        return torch.from_numpy(image).permute(2, 0, 1)

class Resize:
    def __call__(self, image, size):
        image = image.permute(1, 2, 0).numpy()
        image = cv2.resize(image, (size[1], size[0])) # h and w are inverted in cv2 resize
        return torch.from_numpy(image).permute(2, 0, 1)