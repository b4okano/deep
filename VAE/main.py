import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import animation, rc
import cv2
import models

import pickle

BATCH_SIZE = 100


trainval_data = MNIST('./data',
                   train=True,
                   download=True,
                   transform=transforms.ToTensor())

train_size = int(len(trainval_data) * 0.8)
val_size = int(len(trainval_data) * 0.2)
train_data, val_data = torch.utils.data.random_split(trainval_data, [train_size, val_size])

train_loader = DataLoader(dataset=train_data,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=0)

val_loader = DataLoader(dataset=val_data,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        num_workers=0)

print('train data size: ',len(train_data))   #train data size:  48000
print('train iteration number: ',len(train_data)//BATCH_SIZE)   #train iteration number:  480
print('val data size: ',len(val_data))   #val data size:  12000
print('val iteration number: ',len(val_data)//BATCH_SIZE)   #val iteration number:  120

def criterion(predict, target, ave, log_dev):
    bce_loss = F.binary_cross_entropy(predict, target, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + log_dev - ave**2 - log_dev.exp())
    loss = bce_loss + kl_loss
    return loss

z_dim = 2
num_epochs = 20

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = models.VAE(z_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15], gamma=0.1)

history = {'train_loss': [], 'val_loss': [], 'ave': [], 'log_dev': [], 'z': [], 'labels':[]}

for epoch in range(num_epochs):
    model.train()
    for i, (x, labels) in enumerate(train_loader):
        input = x.to(device).view(-1, 28*28).to(torch.float32)
        output, z, ave, log_dev = model(input)

        history['ave'].append(ave)
        history['log_dev'].append(log_dev)
        history['z'].append(z)
        history['labels'].append(labels)
        loss = criterion(output, input, ave, log_dev)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 50 == 0:
            print(f'Epoch: {epoch+1}, loss: {loss: 0.4f}')
        history['train_loss'].append(loss)

    model.eval()
    with torch.no_grad():
        for i, (x, labels) in enumerate(val_loader):
            input = x.to(device).view(-1, 28*28).to(torch.float32)
            output, z, ave, log_dev = model(input)

            loss = criterion(output, input, ave, log_dev)
            history['val_loss'].append(loss)

        print(f'Epoch: {epoch+1}, val_loss: {loss: 0.4f}')

        scheduler.step()





train_loss_tensor = torch.stack(history['train_loss'])
train_loss_np = train_loss_tensor.to('cpu').detach().numpy().copy()
plt.plot(train_loss_np)

val_loss_tensor = torch.stack(history['val_loss'])
val_loss_np = val_loss_tensor.to('cpu').detach().numpy().copy()
plt.plot(val_loss_np)


ave_tensor = torch.stack(history['ave'])
log_var_tensor = torch.stack(history['log_dev'])
z_tensor = torch.stack(history['z'])
labels_tensor = torch.stack(history['labels'])


myfile = open('mydata.pickle', 'wb')
pickle.dump(z_tensor, myfile)
pickle.dump(labels_tensor, myfile)
pickle.dump(ave_tensor, myfile)
myfile.close()

torch.save(model.state_dict(), 'model.pth')
