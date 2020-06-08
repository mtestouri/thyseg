import torch


def dice_(y_pred, y):
    """
    compute the Dice coefficient
    (might not be very accurate) 
    
    parameters
    ----------
    y_pred: tensor
        predictions

    y: tensor
        targets

    c_weights: float array
        the class weights

    returns
    -------
    dice: float
        dice coefficient
    """
    
    smooth = 1.
    y_pred = torch.sigmoid(y_pred).view(len(y_pred), -1)
    y = y.view(len(y), -1)
    intersection = torch.sum(y_pred * y)
    sum_a = torch.sum(y_pred * y_pred)
    sum_b = torch.sum(y * y)
    return ((2. * intersection + smooth) / (sum_a + sum_b + smooth))


def dice(y_pred, y, c_weights=None):
    """
    compute the Dice coefficient
    
    parameters
    ----------
    y_pred: tensor
        predictions tensor of shape: (batch_size, n_channels, height, width)

    y: tensor
        targets tensor of shape: (batch_size, n_channels, height, width)

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

    y: tensor
        targets tensor of shape: (batch_size, n_channels, height, width)

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
        c_weights = torch.softmax(c_weights)

    sum_jacc = 0
    for i in range(y.shape[0]):
        im_a = torch.round(torch.sigmoid(y_pred[i]))
        im_b = torch.round(torch.sigmoid(y[i]))
        
        jacc = 0
        for j in range(y.shape[1]):
            a = im_a[j, :, :]
            b = im_b[j, :, :]
            intersection = torch.relu(a + b - 1)
            union = torch.ceil((a + b) / 2)
            jacc += ((torch.sum(intersection) / torch.sum(union)) * c_weights[j])
        
        sum_jacc += jacc
    return sum_jacc / y.shape[0]