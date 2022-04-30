import torch
import streamlit as st
from utils.torch_utils import select_device
import os
from models.models import *
from pathlib import Path
#taken from this StackOverflow answer: https://stackoverflow.com/a/39225039
import requests

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


def download_data():
    
    path1 = 'trythis/best_overall.pt'

    
    # Local
    # path1 = './data/LastModelResnet50_v2_16.pth.tar'
    # path2 = './data/resnet50_captioning.pt'
    # print("I am here.")
    
    if not os.path.exists(path1):
        #decoder_url = 'wget -O ./best_overall.pt https://drive.google.com/file/d/100_DOjr6dzKaYtcSCOYOUVstKOLSepKe/view?usp=sharing'
        decoder_url = "wget --load-cookies /tmp/cookies.txt 'https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=100_DOjr6dzKaYtcSCOYOUVstKOLSepKe' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=100_DOjr6dzKaYtcSCOYOUVstKOLSepKe' -O trythis/best_overall.pt"
        with st.spinner('done!\nmodel weights were not found, downloading them...'):
            os.system(decoder_url)
    else:
        print("Model [best_overall] is here.")


if __name__ == '__main__':
    st.title("Egyptian Currency Detection")
    

    device = select_device('cpu')


    download_data()
    # get the size of file
    size = os.path.getsize('trythis/best_overall.pt') 
    st.write('Size of file is', size, 'bytes')
    with open('trythis/best_overall.pt') as f:
        lines = f.readlines()
        st.write(lines)
    st.write(os.listdir("./"))
    # Load model
    #model = Darknet('./yolor_p6_custom.cfg', 640)
    #model = Darknet(cfg, imgsz).cuda()

    #model.load_state_dict(torch.load('./best_overall.pt', map_location=device)['model'])