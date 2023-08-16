import cv2 as cv
from ultralytics import YOLO


model = YOLO('/home/michael/Dropbox/MichaelShared/DepthAI/yolov8n_custom/weights/best.pt')

print(model.names)

model.predict(
    source='/home/michael/Desktop/Testing',
    conf=0.4,
    show=False,
    save=True,
    save_crop=False,
    classes=[1, 2, 3]
)

cv.waitKey(0)
