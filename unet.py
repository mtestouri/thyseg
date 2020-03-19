from segmenter import Segmenter
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import math

class UnetSegmenter(Segmenter):
    def __init__(self, input_shape=(512, 512, 3)):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.input_shape = input_shape

    def train(self, x_train, y_train, weights_file=None):
        # dataset
        batch_size = 1
        train_set = ImgSet(x_train, y_train)
        data_loader = DataLoader(dataset=train_set, batch_size=batch_size,
                                 num_workers=2)
        
        # hyper-parameters
        num_epochs = 2
        learning_rate = 0.001

        # model
        model = Unet(self.input_shape)
        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters, lr=learning_rate)
        
        # train loop
        n_iterations = math.ceil(len(train_set)/batch_size)
        for epoch in range(num_epochs):
            for i, (x, y) in enumerate(data_loader):
                # batch
                x = x.to(self.device)
                y = y.to(self.device)

                # forward pass
                #ouputs = model(x)

                # backward pass


                print(f'epoch [{epoch+1}/{num_epochs}], step [{i+1}/{n_iterations}]', end='\r')
        print('')
    
    def predict(self, x_test, weights_file=None):
        pass
    
class Unet(nn.Module):
    def __init__(self,input_shape):
        self.conv1a = nn.Conv2d(3, 64, 3)
        self.conv1b = nn.Conv2d(64, 64, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        #self.conv2a = nn.

class ImgSet(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.n_samples = x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples