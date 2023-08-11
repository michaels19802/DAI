from ultralytics import YOLO


model = YOLO('/home/michael/Projects/DAI/runs/detect/yolov8n_custom/weights/best.pt')

model.export(format='pb')
