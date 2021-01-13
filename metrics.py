import torch


def dice(y_pred, y, c_weights=None):
    """
    compute the Dice coefficient
    
    parameters
    ----------
    y_pred: tensor
        predictions tensor of shape: (batch_size, n_channels, height, width)
        tensor values must be in range [0, 1]

    y: tensor
        targets tensor of shape: (batch_size, n_channels, height, width)
        tensor values must be in range [0, 1]

    c_weights: float array
        the class weights

    returns
    -------
    dice: float
        dice coefficient
    """

    sum_dice = 0
    for i in range(y.shape[0]):
        im_a = y_pred[i].unsqueeze(0)
        im_b = y[i].unsqueeze(0)
        jacc = jaccard(im_a, im_b, c_weights)
        sum_dice += ((2 * jacc) / (1 + jacc))
    return sum_dice / y.shape[0]


def jaccard(y_pred, y, c_weights=None):
    """
    compute the Jaccard index
    
    parameters
    ----------
    y_pred: tensor
        predictions tensor of shape: (batch_size, n_channels, height, width)
        tensor values must be in range [0, 1]

    y: tensor
        targets tensor of shape: (batch_size, n_channels, height, width)
        tensor values must be in range [0, 1]

    c_weights: float array
        class weights

    returns
    -------
    jaccard: float
        jaccard index
    """

    if c_weights is None:
        c_weights = torch.softmax(torch.ones(y.shape[1]), dim=0)
    elif len(c_weights) != y.shape[1]:
        raise ValueError("number of weights must be equal to the number of classes")
    elif torch.sum(c_weights) != 1:
        c_weights = torch.softmax(c_weights, dim=0)

    sum_jacc = 0
    for i in range(y.shape[0]):
        im_a = torch.round(y_pred[i])
        im_b = torch.round(y[i])
        
        jacc = 0
        for j in range(y.shape[1]):
            a = im_a[j, :, :]
            b = im_b[j, :, :]
            intersection = torch.sum(torch.relu(a + b - 1))
            union = torch.sum(torch.ceil((a + b) / 2))
            if union != 0:
                jacc += ((intersection / union) * c_weights[j])
            else:
                jacc += (1 * c_weights[j])
        
        sum_jacc += jacc
    return sum_jacc / y.shape[0]