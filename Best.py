import cv2 as cv
from roboflowoak import RoboflowOak


rf = RoboflowOak(model='barcodes-6kvgt', confidence=0.05, overlap=0.5,
                 version='1', api_key='fAaLtouDRX3lJQdOgyJ1', rgb=True,
                 depth=False, device=None, blocking=True,
                 advanced_config={'wide_fov': True, 'sensor_mode': 'THE_1080_P'})

while True:
    result, frame, raw_frame, depth = rf.detect()
    predictions = result['predictions']

    frame_small = cv.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2), interpolation=cv.INTER_AREA)
    cv.imshow('frame_small', frame_small)

    if cv.waitKey(1) == ord('q'):
        break
