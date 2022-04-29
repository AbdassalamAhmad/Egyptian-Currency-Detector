import cv2
import numpy as np
import streamlit as st
import os
#from gtts import gTTS

#from YOLOR import *

PATH = "test_images"




@st.cache
def download_data():
    
    path1 = './best_overall.pt'

    
    # Local
    # path1 = './data/LastModelResnet50_v2_16.pth.tar'
    # path2 = './data/resnet50_captioning.pt'
    # print("I am here.")
    
    if not os.path.exists(path1):
        decoder_url = 'wget -O ./best_overall.pt https://drive.google.com/file/d/100_DOjr6dzKaYtcSCOYOUVstKOLSepKe/view?usp=sharing'
        
        with st.spinner('done!\nmodel weights were not found, downloading them...'):
            os.system(decoder_url)
    else:
        print("Model [best_overall] is here.")

if __name__ == '__main__':
    # Load class names.
    #names =load_classes(NAMES)
    #print (names)
   
    st.title("Egyptian Currency Detection")
    st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox("Choose the app mode",
                                    ["Demo app", "Detect currency from image"])

    if app_mode == "Demo app":
        st.sidebar.write(" ------ ")
        photos = []
        images = os.listdir(PATH)

        for image in images:
            filepath = os.path.join(PATH, image)
            photos.append(image)
        option = st.sidebar.selectbox(
            'Please select a sample image, then wait for the magic to happen!', photos)
    
        SOURCE = os.path.join(PATH, option)

        # Download the model (.pt) weights from Gdrive using wget
        download_data()
        # Use YOLOR to detect and show img.
        #detect(SOURCE, names)