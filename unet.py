from segmenter import Segmenter
from datasets import ImgDataset, TileDataset
from transforms import Resize, Threshold
from metrics import dice
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms


def pred_pp(thresh=0.5):
    """
    prediction post-processing
    """
    return transforms.Compose([
            Threshold(thresh)
        ])


def dice_loss(y_pred, y, c_weights=None):
    return 1 - dice(y_pred, y, c_weights)


class SegLoss(nn.Module):
    """
    segmentation loss : BCE + Dice

    parameters
    ----------
    c_weights: float array
        class weights used for loss computation
    """
    def __init__(self, c_weights=None):
        super().__init__()
        self._bce_loss = nn.BCELoss()
        self._c_weights = c_weights

    def forward(self, y_pred, y):
        return self._bce_loss(y_pred, y) + dice_loss(y_pred, y, self._c_weights)


class UnetSegmenter(Segmenter):
    """
    U-Net segmenter

    parameters
    ----------
    n_classes: int
        number of classes

    c_weights: float array
        class weights used for loss computation

    init_depth: int
        initial number of filters, the number of filters is doubled at each 
        stages of the U-Net and thus this parameter controls the total 
        number of filters in the network

    device: string
        device to use for segmentation : 'cpu' or 'cuda'
    """

    def __init__(self, n_classes=2, c_weights=None, init_depth=32, device='cuda'):
        super().__init__(n_classes, c_weights, device)
        
        if init_depth < 1:
            raise ValueError("'init_depth' must be greater or equal to 1")
        self._model = Unet(init_depth, n_classes).to(self._device)

    def train(self, folder, n_epochs, tsize=512):
        """
        train the model

        parameters
        ----------
        folder: string
            path to the training set folder

        n_epochs: int
            number of training epochs

        tsize: int
            segmentation tile size
        """

        if n_epochs < 1:
            raise ValueError("'n_epochs' must be greater or equal to 1")

        # training parameters
        learning_rate = 0.0001
        criterion = SegLoss(self._c_weights) # custom loss
        optimizer = torch.optim.Adam(self._model.parameters(), lr=learning_rate)
        
        # inits
        self._model.train()
        tf_resize = Resize()
        dataset = ImgDataset(folder)

        # training loop
        for epoch in range(n_epochs):
            img_count = 0
            sum_loss = 0
            tile_count = 0
            
            for image, mask in dataset:
                tile_dataset = TileDataset(image, mask, tsize=tsize,
                                           mask_merge=(self._n_classes <= 2))
                tile_loader = DataLoader(tile_dataset, batch_size=1)

                for i, (x, y, _) in enumerate(tile_loader):
                    # batch
                    x = x.to(self._device)
                    y = y.to(self._device)

                    # forward pass
                    y_pred = self._model(x)
                    if y_pred.shape != y.shape:
                        y = tf_resize(y, (y_pred.shape[2], y_pred.shape[3]))
                    loss = criterion(y_pred, y)
                    sum_loss += loss.item()

                    # backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # verbose
                    tile_count += 1
                    print("epoch: " + str(epoch+1) + "/" + str(n_epochs)
                          + ", image: " + str(img_count+1) + "/" + str(len(dataset))
                          + ", iteration: " + str(i+1) + "/" + str(len(tile_dataset))
                          + ", avg_loss: " + str(round(sum_loss/tile_count, 4)))
                img_count += 1
        print("training done")


class Unet(nn.Module):
    def __init__(self, init_depth, n_classes):
        super().__init__()
        # encoder
        in_ch = 3
        out_ch = init_depth
        self.conv1 = ConvBlock(in_ch, out_ch)
        in_ch = out_ch
        out_ch = out_ch*2
        self.conv2 = ConvBlock(in_ch, out_ch)
        in_ch = out_ch
        out_ch = out_ch*2
        self.conv3 = ConvBlock(in_ch, out_ch)
        in_ch = out_ch
        out_ch = out_ch*2
        self.conv4 = ConvBlock(in_ch, out_ch)
        in_ch = out_ch
        out_ch = out_ch*2
        self.conv5 = ConvBlock(in_ch, out_ch, pool=False)
        
        # decoder
        in_ch = out_ch
        out_ch = int(out_ch/2)
        self.up_conv6 = UpConvBlock(in_ch, out_ch)
        in_ch = out_ch
        out_ch = int(out_ch/2)
        self.up_conv7 = UpConvBlock(in_ch, out_ch)
        in_ch = out_ch
        out_ch = int(out_ch/2)
        self.up_conv8 = UpConvBlock(in_ch, out_ch)
        in_ch = out_ch
        out_ch = int(out_ch/2)
        self.up_conv9 = UpConvBlock(in_ch, out_ch)
        in_ch = out_ch
        self.conv10 = nn.Conv2d(in_ch, n_classes, 1)
        
        # output
        if n_classes > 1:
            self.output = nn.Softmax(dim=1)
        else:
            self.output = nn.Sigmoid()
    
    def forward(self, x):
        # encoder
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        # decoder
        x = self.up_conv6(x, self.conv4.skip_x)
        x = self.up_conv7(x, self.conv3.skip_x)
        x = self.up_conv8(x, self.conv2.skip_x)
        x = self.up_conv9(x, self.conv1.skip_x)
        x = self.conv10(x)
        return self.output(x)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.skip_x = torch.Tensor()
        
        if pool:
            self.pool = nn.MaxPool2d((2, 2))
        else:
            self.pool = None

    def forward(self, x):
        self.skip_x = self.conv_block(x)
        if self.pool:
            return self.pool(self.skip_x)
        return self.skip_x


class UpConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

    def forward(self, x, skip_x):
        x = self.up(x)
        
        # crop if necessary
        if x.shape!= skip_x.shape:
            skip_x = skip_x[:, :, :x.shape[2], :x.shape[3]] 

        x = torch.cat([x, skip_x], dim=1)
        return F.relu(self.conv2(F.relu(self.conv1(x))))