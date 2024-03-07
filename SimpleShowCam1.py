import depthai as dai
from depthai_sdk import OakCamera


with OakCamera() as oak:
    color = oak.create_camera('color', resolution=dai.ColorCameraProperties.SensorResolution.THE_4_K, fps=10)
    oak.visualize([color], scale=.5, fps=True)
    oak.start(blocking=True)
