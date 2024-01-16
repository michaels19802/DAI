from ultralytics import YOLO


# model = YOLO('yolov8n.pt')
# model = YOLO('yolov8n-seg.pt')
model = YOLO('yolov8n-obb.pt')

model.train(
    data='/home/michael/Ultralytics/Datasets/data.yaml',
    imgsz=640,
    epochs=350,
    batch=16,
    name='yolov8n_custom')
