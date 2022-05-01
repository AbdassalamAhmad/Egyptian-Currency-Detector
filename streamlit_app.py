import cv2
import numpy as np
import streamlit as st
import os
#from gtts import gTTS
import gdown
from YOLOR import *

PATH = "test_images"



# url = "https://drive.google.com/uc?id=100_DOjr6dzKaYtcSCOYOUVstKOLSepKe"
# output = "best_overall.pt"
# gdown.download(url, output, quiet=True)


@st.cache
def download_data():
    
    path1 = './best_overall.pt'
    
    if not os.path.exists(path1):
        url = "https://drive.google.com/uc?id=100_DOjr6dzKaYtcSCOYOUVstKOLSepKe"
        output = "best_overall.pt"
        gdown.download(url, output, quiet=True)
        st.write("Model [best_overall.pt] is being downloaded.")
    else:
        st.write("Model [best_overall.pt] is here.")

if __name__ == '__main__':
    # Load class names.
    names =load_classes(NAMES)
    print (names)
   
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
        st.write(os.listdir("./"))
        # Use YOLOR to detect and show img.
        detect(SOURCE, names)