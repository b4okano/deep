import os
import streamlit as st
import cv2

def show_image(fileN):
    img = cv2.imread(fileN)
    st.image(img)
    return

st.title('show minst images')

path = './number_generated/train'

files = os.listdir(path)

st.title('training data')

num = 4
col = st.columns(num)

i = 0
for fileN in files:
    file =  os.path.join(path, fileN)
    with col[i]:
        img = cv2.imread(file)
        st.image(img, caption=(fileN), use_container_width=True)
    i = (i + 1) % num


st.title('verification data')

path = './number_generated/verify'

files = os.listdir(path)

col2 = st.columns(num)

i = 0
for fileN in files:
    file =  os.path.join(path, fileN)
    with col2[i]:
        img = cv2.imread(file)
        st.image(img, caption=(fileN), use_container_width=True)
    i = (i + 1) % num

