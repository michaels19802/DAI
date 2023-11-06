import requests
import cv2 as cv
import numpy as np


cam_user = 'admin'
cam_password = 'HPPcamera2022'
sess = requests.Session()
sess.auth = (cam_user, cam_password)
url = 'http://192.168.136.49/Streaming/channels/101/picture'


while True:

    res = sess.get(url, stream=True).raw
    arr = np.asarray(bytearray(res.read()), dtype=np.uint8)
    img = cv.imdecode(arr, cv.IMREAD_COLOR)

    img_small = cv.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv.INTER_AREA)

    cv.imshow('img_small', img_small)

    if cv.waitKey(500) & 0xFF == ord('u'):
        break
