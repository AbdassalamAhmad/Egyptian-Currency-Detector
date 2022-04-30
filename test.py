import torch
import streamlit as st
from utils.torch_utils import select_device
import os
from models.models import *



def download_data():
    

    
    # Local
    # path1 = './data/LastModelResnet50_v2_16.pth.tar'
    # path2 = './data/resnet50_captioning.pt'
    # print("I am here.")
    
    
    decoder_url = 'wget -O three_classes_dataset.zip https://drive.google.com/file/d/1f7b-hbT9R3TuxcBBfohpKh3jb_NpBRWF/view?usp=sharing'
    
    with st.spinner('done!\nmodel weights were not found, downloading them...'):
        os.system(decoder_url)




if __name__ == '__main__':
    st.title("Egyptian Currency Detection")
    

    device = select_device('cpu')


    download_data()

    # get the size of file
    size = os.path.getsize('./three_classes_dataset.zip') 
    st.write('Size of file is', size, 'bytes')
    with open('./three_classes_dataset.zip') as f:
        lines = f.readlines()
        st.write(lines)

    # Load model
    #model = Darknet('./yolor_p6_custom.cfg', 640)
    #model = Darknet(cfg, imgsz).cuda()

    #model.load_state_dict(torch.load('./best_overall.pt', map_location=device)['model'])