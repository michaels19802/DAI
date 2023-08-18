from ultralytics import YOLO


# model = YOLO('yolov8n.pt')
model = YOLO('yolov8n-seg.pt')

model.train(
    data='/home/michael/Ultralytics/Data/data.yaml',
    imgsz=640,
    epochs=400,
    batch=8,
    name='yolov8n_custom')
