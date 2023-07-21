import cv2
import time
from roboflowoak import RoboflowOak


rf = RoboflowOak(model='optiscan-lwrhq', confidence=0.4, overlap=0.5,
                 version='3', api_key='ZgLv5crkrzzeRxjigfVo', rgb=True,
                 depth=False, device=None, blocking=True)

while True:
    t0 = time.time()
    result, frame, raw_frame, depth = rf.detect()
    predictions = result['predictions']
    t = time.time() - t0
    print('INFERENCE', 1 / t)
    print('PREDICTIONS', [p.json() for p in predictions])

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break
