import os
import streamlit as st
import cv2

def show_image(fileN):
    img = cv2.imread(fileN)
    st.image(img)
    return

st.title('show learning results')

path = './CNNLearningResult'
acc = 'acc_cnn.png'
loss = 'loss_cnn.png'
path_acc = os.path.join(path, acc)
path_loss = os.path.join(path, loss)

st.title('Accuracy')

show_image(path_acc)

st.title('Loss rate')

show_image(path_loss)

# show wrongly guessed figures

path = './NG_figure_CNN'
files = os.listdir(path)

num = 5
col = st.columns(num)

i = 0
for fileN in files:
    begin = fileN.find('pre-') + len('pre-')
    tail = fileN.find('-ans')
    predict = int(fileN[begin:tail])
    begin = fileN.find('ans-') + len('ans-')
    tail = fileN.find('.jpg')
    ans = int(fileN[begin:tail])
    file =  os.path.join(path, fileN)
    with col[i]:
        img = cv2.imread(file)
        st.image(img, caption=('guess as ' + str(predict)) +\
         ' but answer is ' + str(ans), use_container_width=True)
    i = (i + 1) % num

