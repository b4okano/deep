#import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

import Models as models

import cv2
import os

BATCH_SIZE = 16

train_data = MNIST('./data',
                   train=True,
                   download=True,
                   transform=transforms.ToTensor())

train_loader = DataLoader(train_data,
                          batch_size=BATCH_SIZE,
                          shuffle=True)
print('train data size: ',len(train_data))   #train data size:  60000
print('train iteration number: ',len(train_data)//BATCH_SIZE)   #train iteration number:  3750


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_D = models.Discriminator().to(device)
model_G = models.Generator().to(device)

true_labels = torch.ones(BATCH_SIZE).reshape(BATCH_SIZE, 1).to(device)   #1
fake_labels = torch.zeros(BATCH_SIZE).reshape(BATCH_SIZE, 1).to(device)  #0

criterion = nn.BCELoss()    #BCE (Binary Cross Entropy)

optimizer_D = optim.Adam(model_D.parameters(), lr=0.00001)
optimizer_G = optim.Adam(model_G.parameters(), lr=0.00001)

epoch_num = 100
print_coef = 10
G_train_ratio = 2
train_length = len(train_data)

def calc_acc(pred):
  pred = torch.where(pred > 0.5, 1., 0.)
  acc = pred.sum()/pred.size()[0]
  return acc

history = {'loss_D': [], 'loss_G': [], 'acc_true': [], 'acc_fake': []}
n = 0
m = 0

for epoch in range(epoch_num):
  train_loss_D = 0
  train_loss_G = 0
  train_acc_true = 0
  train_acc_fake = 0

  model_D.train()
  model_G.train()
  for i, data in enumerate(train_loader):
    optimizer_D.zero_grad()
    inputs, labels = data[0].to(device), data[1].to(device)

    # training discriminator 1
    outputs = model_D(inputs, labels)
    loss_true = criterion(outputs, true_labels)
    acc_true = calc_acc(outputs)

    # training discriminator 2
    noise = torch.randn((BATCH_SIZE, 100), dtype=torch.float32).to(device)
    noise_label = torch.from_numpy(np.random.randint(0,10,BATCH_SIZE)).clone().to(device)
    inputs_fake = model_G(noise, noise_label).to(device)
    outputs_fake = model_D(inputs_fake.detach(), noise_label)
    loss_fake = criterion(outputs_fake, fake_labels)
    acc_fake = calc_acc(outputs_fake)
    loss_D = loss_true + loss_fake
    loss_D.backward()
    optimizer_D.step()

    # trainig generator
    for _ in range(G_train_ratio):
      optimizer_G.zero_grad()
      noise = torch.randn((BATCH_SIZE, 100), dtype=torch.float32).to(device)
      noise_label = torch.from_numpy(np.random.randint(0,10,BATCH_SIZE)).clone().to(device)   #1~9のランダムな整数を生成
      inputs_fake = model_G(noise, noise_label).to(device)
      outputs_fake = model_D(inputs_fake, noise_label)
      loss_G = criterion(outputs_fake, true_labels)
      loss_G.backward()
      optimizer_G.step()

    # save history
    train_loss_D += loss_D.item()
    train_loss_G += loss_G.item()
    train_acc_true += acc_true.item()
    train_acc_fake += acc_fake.item()
    n += 1
    history['loss_D'].append(loss_D.item())
    history['loss_G'].append(loss_G.item())
    history['acc_true'].append(acc_true.item())
    history['acc_fake'].append(acc_fake.item())

    if i % ((train_length//BATCH_SIZE)//print_coef) == (train_length//BATCH_SIZE)//print_coef - 1:
      print(f'epoch:{epoch+1}  index:{i+1}  loss_D:{train_loss_D/n:.10f}  loss_G:{train_loss_G/n:.10f}  acc_true:{train_acc_true/n:.10f}  acc_fake:{train_acc_fake/n:.10f}')

      n = 0
      train_loss_D = 0
      train_loss_G = 0
      train_acc_true = 0
      train_acc_fake = 0

print('finish training')

dir_path = './LearningResult'
os.makedirs(dir_path, exist_ok=True)

plt.figure()
plt.plot(history['loss_D'])
plt.xlabel('batch')
plt.ylabel('loss_D')
plt.savefig(os.path.join(dir_path,'loss_D.png'))

plt.figure()
plt.plot(history['loss_G'])
plt.xlabel('batch')
plt.ylabel('loss_G')
plt.savefig(os.path.join(dir_path,'loss_G.png'))


torch.save(model_G.state_dict(), 'Gmodel.pth')
torch.save(model_D.state_dict(), 'Dmodel.pth')
