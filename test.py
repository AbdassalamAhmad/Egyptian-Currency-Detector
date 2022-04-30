import torch
import streamlit as st
from utils.torch_utils import select_device
import os
from models.models import *
import wget


def download_data():
    

    
    # Local
    # path1 = './data/LastModelResnet50_v2_16.pth.tar'
    # path2 = './data/resnet50_captioning.pt'
    # print("I am here.")
    
    
    decoder_url = 'wget -O ./best_overall.pt https://drive.google.com/file/d/100_DOjr6dzKaYtcSCOYOUVstKOLSepKe/view?usp=sharing'
    
    with st.spinner('done!\nmodel weights were not found, downloading them...'):
        os.system(decoder_url)




if __name__ == '__main__':
    st.title("Egyptian Currency Detection")
    

    device = select_device('cpu')


    download_data()

    # get the size of file
    size = os.path.getsize('./best_overall.pt') 
    st.write('Size of file is', size, 'bytes')
    with open('./best_overall.pt') as f:
        lines = f.readlines()
        st.write(lines)

    from IPython.display import FileLink
    st.write(FileLink(r'./best_overall.pt'))
    file_name = wget.download('https://apps.streamlitusercontent.com/abdassalamahmad/egyptian-currency-detector/main/test.py/+/best_overall.pt')
    # Load model
    #model = Darknet('./yolor_p6_custom.cfg', 640)
    #model = Darknet(cfg, imgsz).cuda()

    #model.load_state_dict(torch.load('./best_overall.pt', map_location=device)['model'])