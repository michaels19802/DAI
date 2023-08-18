import cv2 as cv
from ultralytics import YOLO


model = YOLO('best.pt')

print(model.names)

model.predict(
    source='/home/michael/Ultralytics/TrainingImages',
    conf=0.4,
    show=False,
    save=True,
    save_crop=False,
    classes=[0, 1, 2, 3, 4, 5],
    line_width=1,
    retina_masks=True
)

cv.waitKey(0)
