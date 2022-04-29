import time
from pathlib import Path
import streamlit as st
import cv2
import torch
from numpy import random


from utils.datasets import LoadImages
from utils.general import non_max_suppression, apply_classifier, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

from models.models import *
from utils.datasets import *
from utils.general import *

WEIGHTS = ['./best_overall.pt'] # model.pt path[(s)]
SOURCE = '/content/100.jpg' # file or folder to detect objects in it.
IMG_SIZE = 640 # inference size (pixels)
CONF_THRES = 0.4 # object confidence threshold
IOU_THRES = 0.5 # IOU threshold for NMS (boxes near each other)
DEVICE = 'cpu' # cuda device, i.e. 0 or 0,1,2,3 or cpu
VIEW_IMG = True # display results (yes or no)
CLASSES = None # filter by class: --class 0, or --class 0 2 3
AGNOSTIC_NMS = False # class-agnostic NMS
AUGMENT = False # augmented inference
CFG = './yolor_p6_custom.cfg' # *.cfg path
NAMES = './currency.names' # *.names path (object class names of your custom dataset)



def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    # filter removes empty strings (such as last line)
    return list(filter(None, names))


def detect(source = SOURCE, names = NAMES):
    weights, view_img, imgsz, cfg = \
        WEIGHTS, VIEW_IMG, IMG_SIZE, CFG

    # Initialize
    device = select_device(DEVICE)

    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = Darknet(cfg, imgsz).cuda()

    model.load_state_dict(torch.load(weights[0], map_location=device)['model'])
    # model = attempt_load(weights, map_location=device)  # load FP32 model
    # imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load(
            'weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    dataset = LoadImages(source, img_size=imgsz, auto_size=64)

    # Get names and colors
    #names = load_classes(names)
    colors = [[random.randint(0, 255) for _ in range(3)]
              for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    # run once
    _ = model(img.half() if half else img) if device.type != 'cpu' else None
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=AUGMENT)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, CONF_THRES, IOU_THRES, classes=CLASSES, agnostic=AGNOSTIC_NMS)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for ___, det in enumerate(pred):  # detections per image
            p, s, im0 = path, '', im0s

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

            # Print time (inference + NMS)
            print('%s Done inference + NMS. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                st.image(im0)
                #cv2.imwrite('hello.png',im0)
                #cv2.imshow(p, im0)
                #if cv2.waitKey(1) == ord('q'):  # q to quit
                    #break

    print('Done ALL. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    with torch.no_grad():
        detect()
