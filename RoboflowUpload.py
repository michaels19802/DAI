import time
import requests
import cv2 as cv
import numpy as np
from ultralytics import YOLO
from roboflow import Roboflow


cam_user = 'admin'
cam_password = 'HPPcamera2022'
sess = requests.Session()
sess.auth = (cam_user, cam_password)
url = 'http://192.168.136.50/Streaming/channels/101/picture'
rf = Roboflow(api_key='fAaLtouDRX3lJQdOgyJ1')
project = rf.workspace('blue-mmlrh').project('qc-labels')
file_path = '/home/michael/Pictures/RoboflowUpload.png'

model = YOLO('best.pt')

print(model.names)

while True:

    res = sess.get(url, stream=True).raw
    arr = np.asarray(bytearray(res.read()), dtype=np.uint8)
    img = cv.imdecode(arr, cv.IMREAD_COLOR)
    img_small = img[400:-430, 70:-20]
    img_small = cv.resize(img_small, (img_small.shape[1] // 4, img_small.shape[0] // 4), interpolation=cv.INTER_AREA)

    res = model.predict(source=img_small, conf=0.5, classes=[0, 1, 2, 3, 4, 5])

    cv.imshow('Result', res[0].plot())
    cv.imshow('img_small', img_small)

    if cv.waitKey(500) & 0xFF == ord('u'):
        print('Uploading...')
        cv.imwrite(file_path, img_small)
        project.upload(file_path, num_retry_uploads=3)
        print('Uploaded')
        cv.imwrite(f'/home/michael/Ultralytics/TrainingImages/{int(time.time())}.png', img_small)
        print('Saved')
