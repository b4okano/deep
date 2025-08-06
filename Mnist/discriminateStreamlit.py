import sys, os
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import functional as TF
import streamlit as st
from discriminate import Discriminate
import cv2
import tempfile

import torch

torch.classes.__path__ = [] # add this line to manually set it to empty.

parameterFileBody = "params"


def plot_value_array(predictions_array):
    plt.grid(False)
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('blue')
    plt.xlabel(f"Your figure is \"{predicted_label}\" ({100*np.max(predictions_array):.7f}%)")


def predict_image(fileN):

    # prepare data
    img = cv2.imread(fileN)
    img = cv2.resize(img,(28,28))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to gray sclae
    img = TF.to_tensor(img)
    data = []
    data.append(img.detach().numpy())
    data = torch.tensor(np.array(data))

    with torch.no_grad():
        res = disc.discriminate(data)

    plot_value_array(res)
    st.pyplot(fig=plt, use_container_width=True)

    return


def paramFileCheck():
    is_fileCNN = os.path.isfile(parameterFileBody + '-cnn.pth')
    is_fileFCN = os.path.isfile(parameterFileBody + '-fcn.pth')
    if is_fileCNN and is_fileFCN:
        cstat = os.stat(parameterFileBody + '-cnn.pth')
        fstat = os.stat(parameterFileBody + '-fcn.pth')
        if cstat.st_ctime > fstat.st_ctime:
            model = 'cnn'
        else:
            model = 'fcn'
    elif is_fileCNN:
        model = 'cnn'
    elif is_fileFCN:
        model = 'fcn'
    else:
        print('no model exists')
        return None
    return model

#st.set_option('deprecation.showPyplotGlobalUse', False)
st.title('Figre Prediction App')


model= paramFileCheck()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
disc = Discriminate(device, parameterFileBody, model)

image_file = st.file_uploader('Upload a figure image', type=['jpg', 'jpeg', 'png'])

if image_file is not None:
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(image_file.read())
    img = cv2.imread(temp_file.name)
    st.image(img)
    predict_image(temp_file.name)
    temp_file.close()
