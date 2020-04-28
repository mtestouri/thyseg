import sys
from segmenter import Segmenter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import math
import numpy as np
import cv2

def dice_loss(y_pred, y):
    smooth = 1.
    y_pred = torch.sigmoid(y_pred).view(len(y_pred), -1)
    y = y.view(len(y), -1)
    intersection = torch.sum(y_pred * y)
    sum_a = torch.sum(y_pred * y_pred)
    sum_b = torch.sum(y * y)
    return 1 - ((2. * intersection + smooth) / (sum_a + sum_b + smooth))

class SegLoss(nn.Module):
    def __init__(self, mode='both'):
        super().__init__()
        if mode != 'both' and mode != 'bce' and mode != 'dice':
            raise ValueError("invalid mode argument '" + mode + "'")
        self.mode = mode
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, y_pred, y):
        if self.mode == 'bce':
            return self.bce_loss(y_pred, y)
        if self.mode == 'dice':
            return dice_loss(y_pred, y)
        return self.bce_loss(y_pred, y) + dice_loss(y_pred, y)

class UnetSegmenter(Segmenter):
    def init_model(self):
        return Unet()

    def train(self, dataset, n_epochs):
        print("training the model..")
        if n_epochs < 1:
            raise ValueError("the number of epochs must be greater than 0")
        # training parameters
        batch_size = 1
        learning_rate = 0.0001
        criterion = SegLoss() # custom loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        # train loop
        self.model.train()
        dl = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=2)
        n_iterations = math.ceil(len(dataset)/batch_size)
        for epoch in range(n_epochs):
            print('')
            sum_loss = 0
            sum_dice = 0
            for i, (x, y, _) in enumerate(dl):
                # batch
                x = x.to(self.device)
                y = y.to(self.device)
                # forward pass
                y_pred = self.model(x)
                if y_pred.shape != y.shape:
                    y = self.resize(y, (y_pred.shape[2], y_pred.shape[3]))
                loss = criterion(y_pred, y)
                sum_loss += loss.item()
                sum_dice += dice_loss(y_pred, y).item() # compute dice loss
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # verbose
                sys.stdout.write("\033[F") # move cursor up
                sys.stdout.write("\033[K") # clear line
                print("epoch: " + str(epoch+1) + "/" + str(n_epochs)
                      + ", step: " + str(i+1) + "/" + str(n_iterations)
                      + ", avg_loss: " + str(round(sum_loss/(i + 1), 4))
                      + ", avg_dice_loss: " + str(round(sum_dice/(i + 1), 4)))
        print("training done")

    def resize(self, t, size):
        # convert to numpy image and resize
        t = t[:, 1, :, :].cpu().squeeze(0).reshape(t.shape[2], t.shape[3], 1)
        t = torch.cat([t, t, t], dim=2)
        t = torch.round(t*255)
        t = t.numpy()
        t = cv2.resize(t, (size[1], size[0])) # h and w are inverted in cv2
        # convert back to tensor of classes
        t = np.abs(np.round(t/255)[:, :, :2] - (1, 0))
        t = torch.from_numpy(t).permute(2, 0, 1).reshape(1, 2, size[0], size[1])
        return t.to(self.device)

class Unet(nn.Module):
    def __init__(self, init_depth=32, n_classes=2):
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
        return x

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
        if x.shape!= skip_x.shape:
            skip_x = skip_x[:, :, :x.shape[2], :x.shape[3]] # crop if necessary
        x = torch.cat([x, skip_x], dim=1)
        return F.relu(self.conv2(F.relu(self.conv1(x))))
