import os
import streamlit as st
import cv2

def show_image(fileN):
    img = cv2.imread(fileN)
    st.image(img)
    return

st.title('show learning result')

loss = 'lossRate.png'

show_image(loss)

st.title('results')

output = 'output.png'

show_image(output)
