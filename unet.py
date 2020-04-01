from segmenter import Segmenter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import math
import numpy as np

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
        num_epochs = 2
        learning_rate = 0.0001
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        # train loop
        n_iterations = math.ceil(len(train_set)/batch_size)
        for epoch in range(num_epochs):
            for i, (x, y) in enumerate(data_loader):
                # batch
                x = x.to(self.device)
                y = y.to(self.device)
                # forward pass
                y_pred = self.model(x)
                loss = criterion(y_pred, y)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # verbose
                print("epoch: " + str(epoch+1) + "/" + str(num_epochs)
                      + ", step: " + str(i+1) + "/" + str(n_iterations)
                      + ", loss: " + str(round(loss.item(), 4)), end='\r')
            print('')
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
        im_h = x_test.shape[1]
        im_w = x_test.shape[2]
        y_preds = np.empty((len(x_test), im_h, im_w, 2), dtype=np.float32)
        # compute predictions
        for i, x in enumerate(data_loader):
            with torch.no_grad():
                x = x.to(self.device)
                y_pred = F.softmax(self.model(x), dim=1) #TODO sum prob of classes is not 1
                y_preds[i] = y_pred.squeeze(0).reshape(im_h, im_w, 2).cpu().numpy()
            print(f'segmentation: {i+1}/{len(test_set)}', end='\r')
        print('')
        return y_preds

class Unet(nn.Module):
    def __init__(self, init_depth=32, n_classes=2):
        super(Unet, self).__init__()
        # encoder
        self.pool = nn.MaxPool2d((2, 2))
        in_ch = 3
        out_ch = init_depth
        self.conv1 = Conv3x3(in_ch, out_ch)
        in_ch = out_ch
        out_ch = out_ch*2
        self.conv2 = Conv3x3(in_ch, out_ch)
        in_ch = out_ch
        out_ch = out_ch*2
        self.conv3 = Conv3x3(in_ch, out_ch)
        in_ch = out_ch
        out_ch = out_ch*2
        self.conv4 = Conv3x3(in_ch, out_ch)
        in_ch = out_ch
        out_ch = out_ch*2
        self.conv5 = Conv3x3(in_ch, out_ch)
        # decoder
        in_ch = out_ch
        out_ch = int(out_ch/2)
        self.up_conv6 = UpConv2x2(in_ch, out_ch)
        in_ch = out_ch
        out_ch = int(out_ch/2)
        self.up_conv7 = UpConv2x2(in_ch, out_ch)
        in_ch = out_ch
        out_ch = int(out_ch/2)
        self.up_conv8 = UpConv2x2(in_ch, out_ch)
        in_ch = out_ch
        out_ch = int(out_ch/2)
        self.up_conv9 = UpConv2x2(in_ch, out_ch)
        in_ch = out_ch
        self.conv10 = Conv1x1(in_ch, n_classes)
    
    def forward(self, x):
        # encoder
        x1 = self.conv1(x)
        x = self.pool(x1)
        x2 = self.conv2(x)
        x = self.pool(x2)
        x3 = self.conv3(x)
        x = self.pool(x3)
        x4 = self.conv4(x)
        x = self.pool(x4)
        x = self.conv5(x)
        # decoder
        x = self.up_conv6(x, x4)
        x = self.up_conv7(x, x3)
        x = self.up_conv8(x, x2)
        x = self.up_conv9(x, x1)
        x = self.conv10(x)
        return x

class Conv3x3(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Conv3x3, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        #self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        return F.relu(self.conv2(F.relu(self.conv1(x))))

class Conv1x1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.conv(x)

class UpConv2x2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpConv2x2, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

    def forward(self, x, skip_x):
        x = self.up(x)
        x = torch.cat([x, skip_x], dim=1)
        return F.relu(self.conv2(F.relu(self.conv1(x))))

class ImgSet(Dataset):
    def __init__(self, x, y=None):
        self.n_samples = x.shape[0]
        # images
        x = x.reshape(x.shape[0], 3, x.shape[1], x.shape[2])
        self.x = torch.from_numpy(x)#/255
        #tf = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25))
        #for i in range(len(self.x)):
        #    self.x[i] = tf(self.x[i])
        # masks
        if y is not None:
            y = y.reshape(y.shape[0], 2, y.shape[1], y.shape[2])
            self.y = torch.from_numpy(y)
        else:
            self.y = None
        
    def __getitem__(self, index):
        if self.y is not None:
            return self.x[index], self.y[index]
        return self.x[index]

    def __len__(self):
        return self.n_samples