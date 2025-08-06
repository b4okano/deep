import os
import streamlit as st
import cv2

def show_image(fileN):
    img = cv2.imread(fileN)
    st.image(img)
    return

st.title('show learning results')

path = './LearningResult'
lossd = 'loss_D.png'
lossg = 'loss_G.png'
path_lossd = os.path.join(path, lossd)
path_lossg = os.path.join(path, lossg)

st.title('loss D')

show_image(path_lossd)

st.title('Loss G')

show_image(path_lossg)