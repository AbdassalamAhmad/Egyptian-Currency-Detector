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





if __name__ == '__main__':
    st.title("Egyptian Currency Detection")
    

    device = select_device('cpu')


    file_id = '100_DOjr6dzKaYtcSCOYOUVstKOLSepKe'
    destination = 'model/best_overall.pt'
    download_file_from_google_drive(file_id, destination)

    # get the size of file
    size = os.path.getsize('model/best_overall.pt') 
    st.write('Size of file is', size, 'bytes')
    with open('model/best_overall.pt') as f:
        lines = f.readlines()
        st.write(lines)
    st.write(os.listdir("./model"))
    # Load model
    #model = Darknet('./yolor_p6_custom.cfg', 640)
    #model = Darknet(cfg, imgsz).cuda()

    #model.load_state_dict(torch.load('./best_overall.pt', map_location=device)['model'])