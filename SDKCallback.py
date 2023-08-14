import cv2 as cv
import depthai as dai
from depthai_sdk import OakCamera
from depthai_sdk.visualize.visualizer_helper import FramePosition, VisualizerHelper


def callback(packet):
    visualizer = packet.visualizer
    print('Detections:', packet.img_detections.detections)
    VisualizerHelper.print(packet.frame, str(len(packet.img_detections.detections)), FramePosition.BottomRight)
    frame = visualizer.draw(packet.frame)
    frame_small = cv.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2), interpolation=cv.INTER_AREA)
    cv.imshow('Visualizer', frame_small)


with OakCamera() as oak:
    color = oak.create_camera('color')
    color.config_color_camera(scene_mode=dai.CameraControl.SceneMode.BARCODE,
                              awb_mode=dai.CameraControl.AutoWhiteBalanceMode.DAYLIGHT)
    nn = oak.create_nn('/home/michael/Dropbox/MichaelShared/DepthAI/result/best.json', color)
    oak.visualize(nn, fps=True, callback=callback)
    oak.start(blocking=True)
