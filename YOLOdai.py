import cv2
import depthai as dai
import numpy as np
import time


labels = ['2D barcode', 'Datamatrix', 'QR code']

pipeline = dai.Pipeline()

camRgb = pipeline.create(dai.node.ColorCamera)
detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
xoutRgb = pipeline.create(dai.node.XLinkOut)
nnOut = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName('rgb')
nnOut.setStreamName('nn')

camRgb.setPreviewSize(640, 640)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
camRgb.initialControl.setEffectMode(dai.RawCameraControl.EffectMode.MONO)

detectionNetwork.setConfidenceThreshold(.1)
detectionNetwork.setNumClasses(3)
detectionNetwork.setCoordinateSize(4)
detectionNetwork.setAnchors([])
detectionNetwork.setAnchorMasks({})
detectionNetwork.setIouThreshold(.5)
detectionNetwork.setBlobPath('/home/michael/Dropbox/MichaelShared/DepthAI/result/best_openvino_2022.1_6shave.blob')
detectionNetwork.setNumInferenceThreads(2)
detectionNetwork.input.setBlocking(False)

camRgb.preview.link(detectionNetwork.input)
detectionNetwork.passthrough.link(xoutRgb.input)
detectionNetwork.out.link(nnOut.input)

with dai.Device(pipeline) as device:
    qRgb = device.getOutputQueue(name='rgb', maxSize=4, blocking=False)
    qDet = device.getOutputQueue(name='nn', maxSize=4, blocking=False)

    frame = None
    detections = []
    startTime = time.monotonic()
    counter = 0
    color2 = (255, 255, 255)

    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def displayFrame(name, frame, detections):
        color = (255, 0, 0)
        for detection in detections:
            bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            cv2.putText(frame, labels[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f'{int(detection.confidence * 100)}%', (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        cv2.imshow(name, frame)

    while True:
        inRgb = qRgb.get()
        inDet = qDet.get()

        if inRgb is not None:
            frame = inRgb.getCvFrame()
            cv2.putText(frame, 'NN fps: {:.2f}'.format(counter / (time.monotonic() - startTime)),
                        (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color2)

        if inDet is not None:
            detections = inDet.detections
            counter += 1

        if frame is not None:
            displayFrame('rgb', frame, detections)

        if cv2.waitKey(1) == ord('q'):
            break
