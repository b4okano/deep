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
import os

import pickle

BATCH_SIZE = 100

z_dim = 2
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
params = torch.load('model.pth', map_location=torch.device('cpu'))
model = models.VAE(z_dim).to('cpu')
model.load_state_dict(params)
outputdir='output'
os.makedirs(outputdir, exist_ok=True)

def make_image(label):
    x_mean = np.mean(ave_np[batch_num:,:,0][labels_np[batch_num:,:] == label])
    y_mean = np.mean(ave_np[batch_num:,:,1][labels_np[batch_num:,:] == label])
    z = torch.tensor([x_mean, y_mean], dtype = torch.float32)

    plt.figure(figsize=[10, 10])
    output = model.to('cpu').decoder(z)
    np_output = output.to('cpu').detach().numpy().copy()
    np_image = np.reshape(np_output, (28, 28))
    plt.imshow(np_image, cmap='gray')
    fname = os.path.join(outputdir, 'fig-label-'+str(label)+'.png')
    plt.savefig(fname)
    return z




myfile = open('mydata.pickle', 'rb')
z_tensor = pickle.load(myfile)
labels_tensor = pickle.load(myfile)
ave_tensor = pickle.load(myfile)
myfile.close()

z_np = z_tensor.to('cpu').detach().numpy().copy()
labels_np = labels_tensor.to('cpu').detach().numpy().copy()
ave_np = ave_tensor.to('cpu').detach().numpy().copy()

cmap_keyword = 'tab10'
cmap = plt.get_cmap(cmap_keyword)

batch_num =10
plt.figure(figsize=[10, 10])
for label in range(10):
    x = z_np[:batch_num,:,0][labels_np[:batch_num,:] == label]
    y = z_np[:batch_num,:,1][labels_np[:batch_num,:] == label]
    plt.scatter(x, y, color=cmap(label/9), label=label, s=15)
    plt.annotate(label, xy=(np.mean(x),np.mean(y)),size=20, color='black')
plt.legend(loc='upper left')
fname1 = os.path.join(outputdir, 'fig1.png')
plt.savefig(fname1)

batch_num = 9580
plt.figure(figsize=[10,10])
for label in range(10):
    x = z_np[batch_num:,:,0][labels_np[batch_num:,:] == label]
    y = z_np[batch_num:,:,1][labels_np[batch_num:,:] == label]
    plt.scatter(x, y, color=cmap(label/9), label=label, s=15)
    plt.annotate(label, xy=(np.mean(x),np.mean(y)),size=20, color='black')
plt.legend(loc='upper left')
fname2 = os.path.join(outputdir, 'fig2.png')
plt.savefig(fname2)


z0 = make_image(0)
z1 = make_image(1)
z2 = make_image(2)
z3 = make_image(3)
z4 = make_image(4)

z7 = make_image(7)
z8 = make_image(8)


def plot(frame):
    plt.cla()
    if frame < 100:
      z_0to8 = ((99 - frame) * z0 +  frame * z1) / 99
    elif frame < 200:
      z_0to8 = ((199 - frame) * z1 +  (frame-100) * z8) / 99
    else:
      z_0to8 = ((299 - frame) * z8 +  (frame-200) * z7) / 99
    output = model.decoder(z_0to8)
    np_output = output.detach().numpy().copy()
    np_image = np.reshape(np_output, (28, 28))
    plt.imshow(np_image, cmap='gray')
    plt.xticks([]);plt.yticks([])
    plt.title('frame={}'.format(frame))

fig = plt.figure(figsize=(4, 4))
ani = animation.FuncAnimation(fig, plot, frames=299, interval=100)
#rc('animation', html='jshtml')
#ani.save('output.gif', writer='imagemagick')
fname3 = os.path.join(outputdir, 'output.gif')
ani.save(fname3, writer='pillow')
