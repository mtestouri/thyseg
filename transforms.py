import torch
import cv2

class Normalize:
    def __call__(self, image):
        for i in range(len(image)):
            image[i] = (image[i] - torch.mean(image[i]))/torch.std(image[i])
        return image

class Smoothing:
    def __init__(self, ksize):
        if ksize < 1:
            raise ValueError("kernel size must be greater or equal to 1")
        self.ksize = ksize

    def __call__(self, image):
        image = image.permute(1, 2, 0).numpy()
        image = cv2.blur(image, (self.ksize, self.ksize))
        return torch.from_numpy(image).permute(2, 0, 1)

class Threshold:
    def __init__(self, thresh):
        if thresh < 0 or thresh > 1:
            raise ValueError("threshold must belong to [0,1]")
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