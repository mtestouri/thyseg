import torch

# TODO dice > 1 et pb avec jaccard ??

def dice(y_pred, y):
    smooth = 1.
    y_pred = torch.sigmoid(y_pred).view(len(y_pred), -1)
    y = y.view(len(y), -1)
    intersection = torch.sum(y_pred * y)
    sum_a = torch.sum(y_pred * y_pred)
    sum_b = torch.sum(y * y)
    return ((2. * intersection + smooth) / (sum_a + sum_b + smooth))

def jaccard(y_pred, y):
    d = dice(y_pred, y)
    return d / (2 - d)