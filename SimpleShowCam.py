import depthai as dai
from depthai_sdk import OakCamera


with OakCamera(rotation=180) as oak:
    color = oak.create_camera('color', resolution=dai.ColorCameraProperties.SensorResolution.THE_12_MP, fps=7)
    color.config_color_camera(awb_mode=dai.CameraControl.AutoWhiteBalanceMode.TWILIGHT,
                              effect_mode=dai.CameraControl.EffectMode.WHITEBOARD,
                              sharpness=4)
    oak.visualize([color], scale=.25, fps=True)
    oak.start(blocking=True)
