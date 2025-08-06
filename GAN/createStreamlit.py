from matplotlib import pyplot as plt
import numpy as np
import torch

import Models as models
import streamlit as st

import cv2
import os


def change():
  label_np = np.random.randint(theNumber ,theNumber + 1 ,100)

  with torch.no_grad():
    noise = torch.randn((100, 100), dtype=torch.float32)
    noise.to(device)
    noise_label = torch.from_numpy(label_np).clone()
    noise_label.to(device)
    model_G.to(device)
    syn_image = model_G(noise, noise_label)


  for i in range(10):
    plt.figure()
    plt.imshow(syn_image[i,0,:,:], cmap='gray')
    plt.savefig('tmp.png')

    with cols[i]:
      img = cv2.imread('tmp.png')
      st.image(img)


torch.classes.__path__ = [] # add this line to manually set it to empty.

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
model_G = models.Generator().to(device)

params = torch.load('Gmodel.pth', map_location=device)
model_G.load_state_dict(params)
model_G.to(device)

st.title('GAN Generated Number App')

number_options = range(10)

theNumber = st.selectbox('select the number to generate:', number_options)

st.button('create', on_click=change)
cols = st.columns(10)