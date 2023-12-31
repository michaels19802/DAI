import depthai as dai
from depthai_sdk import OakCamera


with OakCamera() as oak:
    color = oak.create_camera('color')

    color.config_color_camera(scene_mode=dai.CameraControl.SceneMode.BARCODE,
                              awb_mode=dai.CameraControl.AutoWhiteBalanceMode.DAYLIGHT)

    nn = oak.create_nn('/home/michael/Dropbox/MichaelShared/DepthAI/result/best.json', color)

    visualizer = oak.visualize(nn, fps=True)

    oak.start(blocking=True)
