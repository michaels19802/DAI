import depthai as dai
from depthai_sdk import OakCamera


with OakCamera(rotation=180) as oak:
    color = oak.create_camera('color', resolution='1080p', fps=15)
    color.config_color_camera(awb_mode=dai.CameraControl.AutoWhiteBalanceMode.TWILIGHT,
                              effect_mode=dai.CameraControl.EffectMode.WHITEBOARD,
                              sharpness=4)
    left = oak.create_camera('left', resolution='800p', fps=15)
    oak.visualize([color, left], fps=True)
    oak.start(blocking=True)
