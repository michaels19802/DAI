import cv2 as cv
from ultralytics import YOLO


model = YOLO('/home/michael/Dropbox/MichaelShared/DepthAI/yolov8n_custom/weights/best.pt')

print(model.names)

model.predict(
    source='/home/michael/Desktop/img_small_screenshot_16.08.2023.png',
    conf=0.2,
    show=True,
    save=False,
    save_crop=False,
    classes=None
)

cv.waitKey(0)
