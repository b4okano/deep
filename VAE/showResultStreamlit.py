import os
import streamlit as st
import cv2

def show_image(fileN):
    img = cv2.imread(fileN)
    st.image(img)
    return

st.title('show outputs')

path = 'output'
fig1 = 'fig1.png'
fig2 = 'fig2.png'
figl0 = 'fig-label-0.png'
figl1 = 'fig-label-1.png'
figl2 = 'fig-label-2.png'
figl3 = 'fig-label-3.png'
figl4 = 'fig-label-4.png'
figl7 = 'fig-label-7.png'
figl8 = 'fig-label-8.png'
gif = 'output.gif'

fig1path = os.path.join(path, fig1)

st.title('early stage')

show_image(fig1path)

st.title('last stage')

fig2path = os.path.join(path, fig2)

show_image(fig2path)

st.title('genarated figutes')

num = 5
col = st.columns(num)

figl0path = os.path.join(path, figl0)
figl1path = os.path.join(path, figl1)
figl2path = os.path.join(path, figl2)
figl3path = os.path.join(path, figl3)
figl4path = os.path.join(path, figl4)
figl7path = os.path.join(path, figl7)
figl8path = os.path.join(path, figl8)
gifpath  = os.path.join(path, gif)

i = 0
with col[i]:
    img = cv2.imread(figl0path)
    st.image(img, caption=('figure 0'), use_container_width=True)
    i = (i + 1) % num
with col[i]:
    img = cv2.imread(figl1path)
    st.image(img, caption=('figure 1'), use_container_width=True)
    i = (i + 1) % num
with col[i]:
    img = cv2.imread(figl2path)
    st.image(img, caption=('figure 2'), use_container_width=True)
    i = (i + 1) % num
with col[i]:
    img = cv2.imread(figl3path)
    st.image(img, caption=('figure 3'), use_container_width=True)
    i = (i + 1) % num
with col[i]:
    img = cv2.imread(figl4path)
    st.image(img, caption=('figure 4'), use_container_width=True)
    i = (i + 1) % num
with col[i]:
    img = cv2.imread(figl7path)
    st.image(img, caption=('figure 7'), use_container_width=True)
    i = (i + 1) % num
with col[i]:
    img = cv2.imread(figl8path)
    st.image(img, caption=('figure 8'), use_container_width=True)
    i = (i + 1) % num

with open(gifpath, 'rb') as file:
    img_data = file.read()

st.download_button(
    label='Download gif image',
    data=img_data,
    file_name='output.gif',
    mime='image/gif'
)