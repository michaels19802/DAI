import depthai as dai
from depthai_sdk import OakCamera


with OakCamera() as oak:
    color = oak.create_camera('color', resolution=dai.ColorCameraProperties.SensorResolution.THE_1080_P, fps=20)
    oak.visualize([color], fps=True)
    oak.start(blocking=True)
