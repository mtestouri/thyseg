import sys
from segmenter import Segmenter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import math
import numpy as np

def dice_loss(y_pred, y):
    smooth = 1.
    y_pred = torch.sigmoid(y_pred).view(len(y_pred), -1)
    y = y.view(len(y), -1)
    intersection = torch.sum(y_pred * y)
    sum_a = torch.sum(y_pred * y_pred)
    sum_b = torch.sum(y * y)
    return 1 - ((2. * intersection + smooth) / (sum_a + sum_b + smooth))

class UnetSegmenter(Segmenter):
    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.model = Unet().to(self.device)

    def train(self, x_train, y_train, model_file=None):
        # create train set
        batch_size = 1
        train_set = ImgSet(x_train, y_train)
        data_loader = DataLoader(dataset=train_set, batch_size=batch_size,
                                 num_workers=2)
        # training parameters
        num_epochs = 8
        learning_rate = 0.0001
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        # train loop
        n_iterations = math.ceil(len(train_set)/batch_size)
        for epoch in range(num_epochs):
            print('')
            sum_loss = 0
            sum_dice = 0
            for i, (x, y) in enumerate(data_loader):
                # batch
                x = x.to(self.device)
                y = y.to(self.device)
                # forward pass
                y_pred = self.model(x)
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
                print("epoch: " + str(epoch+1) + "/" + str(num_epochs)
                      + ", step: " + str(i+1) + "/" + str(n_iterations)
                      + ", avg_loss: " + str(round(sum_loss/(i + 1), 4))
                      + ", avg_dice_loss: " + str(round(sum_dice/(i + 1), 4)))
        # save model
        if model_file:
            torch.save(self.model.state_dict(), model_file)
    
    def predict(self, x_test, model_file=None):
        # load model
        if model_file:
            self.model.load_state_dict(torch.load(model_file))
        # create test set
        test_set = ImgSet(x_test)
        data_loader = DataLoader(dataset=test_set, batch_size=1, num_workers=2)
        # init predicted masks
        y_preds = np.empty((len(x_test), x_test.shape[1], x_test.shape[2], 2), 
                           dtype=np.float32)
        # compute predictions
        for i, x in enumerate(data_loader):
            with torch.no_grad():
                x = x.to(self.device)
                y_pred = torch.round(torch.sigmoid(self.model(x)))
                y_preds[i] = y_pred.permute(0, 2, 3, 1).squeeze(0).cpu().numpy()
            print(f'segmentation: {i+1}/{len(test_set)}', end='\r')
        print('')
        return y_preds

class Unet(nn.Module):
    def __init__(self, init_depth=32, n_classes=2):
        super(Unet, self).__init__()
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
        super(ConvBlock, self).__init__()
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
        super(UpConvBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

    def forward(self, x, skip_x):
        x = self.up(x)
        x = torch.cat([x, skip_x], dim=1)
        return F.relu(self.conv2(F.relu(self.conv1(x))))

class ImgSet(Dataset):
    def __init__(self, x, y=None):
        # images
        self.n_samples = len(x)
        self.x = torch.from_numpy(x).float().permute(0, 3, 1, 2)
        # masks
        if y is not None:
            self.y = torch.from_numpy(y).float().permute(0, 3, 1, 2)
        else:
            self.y = None
        
    def __getitem__(self, index):
        if self.y is not None:
            return self.x[index], self.y[index]
        return self.x[index]

    def __len__(self):
        return self.n_samples