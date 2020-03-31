from segmenter import Segmenter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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
        # dataset
        batch_size = 1
        train_set = ImgSet(x_train, y_train)
        data_loader = DataLoader(dataset=train_set, batch_size=batch_size,
                                 num_workers=2)
        # training parameters
        num_epochs = 2
        learning_rate = 0.001
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
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
        self.model = self.model.to('cpu') #TODO fix mem problem with CUDA
        # predictions
        im_h = x_test.shape[1]
        im_w = x_test.shape[2]
        y_preds = np.empty((len(x_test), im_h, im_w, 2), dtype=np.float32)
        for i, x in enumerate(x_test):
            with torch.no_grad():
                x = torch.from_numpy(x.reshape(1, 3, im_h, im_w))
                y_pred = self.model(x)
                y_pred = F.softmax(y_pred, dim=1) #TODO sum not 1 !!
                y_pred = y_pred.squeeze(0).reshape(im_h, im_w, 2)
                y_preds[i] = y_pred.cpu().numpy()
            print(f'segmentation: {i+1}/{len(x_test)}', end='\r')
        print('')
        self.model = self.model.to('cuda')
        return y_preds

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        # encoder
        self.conv1 = Conv3x3(3, 32)
        self.conv2 = Conv3x3(32, 64)
        self.conv3 = Conv3x3(64, 128)
        self.conv4 = Conv3x3(128, 256)
        self.conv5 = Conv3x3(256, 512, use_pool=False)
        # decoder
        self.up_conv6 = UpConv2x2(512, 256)
        self.up_conv7 = UpConv2x2(256, 128)
        self.up_conv8 = UpConv2x2(128, 64)
        self.up_conv9 = UpConv2x2(64, 32)
        self.conv10 = Conv1x1(32, 2)
    
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

class Conv3x3(nn.Module):
    def __init__(self, in_ch, out_ch, use_pool=True):
        super(Conv3x3, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip_x = torch.Tensor
        self.use_pool = use_pool
        if self.use_pool:
            self.pool = nn.MaxPool2d((2, 2))

    def forward(self, x):
        self.skip_x = F.relu(self.conv2(F.relu(self.conv1(x))))
        if self.use_pool:
            return self.pool(self.skip_x)
        return self.skip_x

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
        x = torch.cat([x, skip_x], 1)
        return F.relu(self.conv2(F.relu(self.conv1(x))))

class ImgSet(Dataset):
    def __init__(self, x, y):
        x = x.reshape(x.shape[0], 3, x.shape[1], x.shape[2])
        y = y.reshape(y.shape[0], 2, y.shape[1], y.shape[2])
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.n_samples = x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples