from segmenter import Segmenter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np

class UnetSegmenter(Segmenter):
    def __init__(self, input_shape=(512, 512, 3)):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.input_shape = input_shape
        self.model = Unet(self.input_shape).to(self.device) #TODO input shape required ?

    def train(self, x_train, y_train, model_file=None):
        # dataset
        batch_size = 1
        train_set = ImgSet(x_train, y_train)
        data_loader = DataLoader(dataset=train_set, batch_size=batch_size,
                                 num_workers=2)
        # training parameters
        num_epochs = 2
        learning_rate = 0.001
        criterion = nn.BCELoss()
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
                loss = criterion(y_pred/torch.max(y_pred), y.float()) #TODO normalization required ?
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # verbose
                print(f'epoch: {epoch+1}/{num_epochs}, step: {i+1}/{n_iterations}, loss: {loss.item():.4f}', end='\r')
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
        y_preds = np.empty((len(x_test), 512, 512, 3), dtype=np.float32)
        #x_test = x_test.reshape(len(x_test), 3, 512, 512) #TODO use input shape instead ?
        #x_test = torch.from_numpy(x_test)
        for i, x in enumerate(x_test):
            x = torch.from_numpy(x.reshape(1, 3, 512, 512)) #TODO use input shape instead ?
            y_pred = self.model(x)
            y_preds[i] = y_pred.detach().numpy().reshape(512, 512, 3) #TODO use input shape instead ?
            print(i, end='\r') 
        self.model = self.model.to('cuda')
        print(y_preds.shape)
        return y_preds

class Unet(nn.Module):
    def __init__(self, input_shape):
        super(Unet, self).__init__()
        # encoder
        self.conv1a = nn.Conv2d(3, 32, 3, padding=1)
        self.conv1b = nn.Conv2d(32, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d((2, 2))
        self.conv2a = nn.Conv2d(32, 64, 3, padding=1)
        self.conv2b = nn.Conv2d(64, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d((2, 2))
        self.conv3a = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3b = nn.Conv2d(128, 128, 3, padding=1)
        self.pool3 = nn.MaxPool2d((2, 2))
        self.conv4a = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4b = nn.Conv2d(256, 256, 3, padding=1)
        self.pool4 = nn.MaxPool2d((2, 2))
        self.conv5a = nn.Conv2d(256, 512, 3, padding=1)
        self.conv5b = nn.Conv2d(512, 512, 3, padding=1)
        # decoder
        self.up6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv6a = nn.Conv2d(512, 256, 3, padding=1)
        self.conv6b = nn.Conv2d(256, 256, 3, padding=1)
        self.up7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7a = nn.Conv2d(256, 128, 3, padding=1)
        self.conv7b = nn.Conv2d(128, 128, 3, padding=1)
        self.up8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8a = nn.Conv2d(128, 64, 3, padding=1)
        self.conv8b = nn.Conv2d(64, 64, 3, padding=1)
        self.up9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9a = nn.Conv2d(64, 32, 3, padding=1)
        self.conv9b = nn.Conv2d(32, 32, 3, padding=1)
        self.conv9c = nn.Conv2d(32, 3, 1)
    
    def forward(self, x):
        # encoder
        u1 = F.relu(self.conv1b(F.relu(self.conv1a(x))))
        #print(u1.shape)
        x = self.pool1(u1)
        u2 = F.relu(self.conv2b(F.relu(self.conv2a(x))))
        #print(u2.shape)
        x = self.pool2(u2)
        u3 = F.relu(self.conv3b(F.relu(self.conv3a(x))))
        #print(u3.shape)
        x = self.pool3(u3)
        u4 = F.relu(self.conv4b(F.relu(self.conv4a(x))))
        #print(u4.shape)
        x = self.pool4(u4)
        x = F.relu(self.conv5b(F.relu(self.conv5a(x))))
        #print(x.shape)
        # decoder
        x = self.up6(x)
        #print(x.shape)
        x = torch.cat([x, u4], 1)
        x = F.relu(self.conv6b(F.relu(self.conv6a(x))))
        x = self.up7(x)
        #print(x.shape)
        x = torch.cat([x, u3], 1)
        x = F.relu(self.conv7b(F.relu(self.conv7a(x))))
        x = self.up8(x)
        #print(x.shape)
        x = torch.cat([x, u2], 1)
        x = F.relu(self.conv8b(F.relu(self.conv8a(x))))
        x = self.up9(x)
        #print(x.shape)
        x = torch.cat([x, u1], 1)
        x = F.relu(self.conv9b(F.relu(self.conv9a(x))))
        x = F.relu(self.conv9c(x))
        return x

class ImgSet(Dataset):
    def __init__(self, x, y):
        x = x.reshape(len(x), 3, 512, 512) #TODO use input shape instead ?
        y = y.reshape(len(y), 3, 512, 512)
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.n_samples = x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples