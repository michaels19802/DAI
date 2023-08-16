from ultralytics import YOLO
import cv2


model = YOLO('/home/michael/Dropbox/MichaelShared/DepthAI/yolov8n_custom/weights/best.pt')

model.predict(
    source='/home/michael/Desktop/img_small_screenshot_16.08.2023.png',
    conf=0.2,
    show=True,
    save_crop=True
)

cv2.waitKey(0)
