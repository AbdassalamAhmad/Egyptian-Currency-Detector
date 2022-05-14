import cv2
import numpy as np
import streamlit as st
import os

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

@st.cache
def load_model():
    model = Darknet(CFG, IMG_SIZE)
    return model


if __name__ == '__main__':
    # Load class names.
    names =load_classes(NAMES)
    print (names)
   
    st.title("Egyptian Currency Detection")
    st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox("Choose the app mode",
                                    ["Demo app", "Detect currency using camera"])

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
        model = load_model()
        
        # Use YOLOR to detect and show img.
        detect(SOURCE, names, model)
    
    elif app_mode == "Detect currency using camera":
        st.subheader("After taking a picture you will get your photo with\
         the detected currency bounding box and a sound of the banknote's name")
        # Load image.
        picture = st.camera_input("Take a picture")
        if picture is not None:
            # To read image file buffer with OpenCV:
            bytes_data = picture.getvalue()
            img0 = cv2.imdecode(np.frombuffer(
                bytes_data, np.uint8), cv2.IMREAD_COLOR)
            
            img = letterbox(img0, new_shape=640, auto_size=32)[0]

            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)


            img = torch.from_numpy(img).to('cpu')
            img = img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            model = load_model()
            #model = Darknet(cfg, imgsz).cuda()

            model.load_state_dict(torch.load(WEIGHTS[0], map_location='cpu')['model'])
            # model = attempt_load(weights, map_location=device)  # load FP32 model
            # imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
            model.to('cpu').eval()
            pred = model(img, augment=AUGMENT)[0]

            # Apply NMS
            pred = non_max_suppression(
                pred, CONF_THRES, IOU_THRES, classes=CLASSES, agnostic=AGNOSTIC_NMS)
            t2 = time_synchronized()
            colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

            # Process detections
            for ___, det in enumerate(pred):  # detections per image
                s, im0 = '', img0

                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, names[int(c)])  # add to string

                    # Write results
                    for *xyxy, conf, cls in det:
                        # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label,
                                    color=colors[int(cls)], line_thickness=3)
                        to_be_said = '%s' % (names[int(cls)])
                        myobj = gTTS(text=to_be_said, lang='en', slow=False)
                        myobj.save("detected_currency.mp3")
                        audio_file = open('detected_currency.mp3', 'rb')
                        audio_bytes = audio_file.read()
                        st.audio(audio_bytes, format='audio/ogg', start_time=0)
                # Print time (inference + NMS)
                print('%s Done inference + NMS. (%.3fs)' % (s, t2 - t1))


                im_rgb = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
                st.image(im_rgb)

                
