import requests
import cv2 as cv
import numpy as np
from roboflow import Roboflow


cam_user = 'admin'
cam_password = 'HPPcamera2022'
sess = requests.Session()
sess.auth = (cam_user, cam_password)
url = 'http://192.168.136.50/Streaming/channels/101/picture'
rf = Roboflow(api_key='fAaLtouDRX3lJQdOgyJ1')
project = rf.workspace('blue-mmlrh').project('qc-labels')
file_path = '/home/michael/Pictures/RoboflowUpload.png'

while True:

    res = sess.get(url, stream=True).raw
    arr = np.asarray(bytearray(res.read()), dtype=np.uint8)
    img = cv.imdecode(arr, cv.IMREAD_COLOR)
    img_small = img[400:-430, 70:-20]
    img_small = cv.resize(img_small, (img_small.shape[1] // 4, img_small.shape[0] // 4), interpolation=cv.INTER_AREA)

    cv.imshow('img_small', img_small)

    if cv.waitKey(2000) & 0xFF == ord('u'):
        print('Uploading...')
        cv.imwrite(file_path, img_small)
        project.upload(file_path, num_retry_uploads=3)
        print('Uploaded')
