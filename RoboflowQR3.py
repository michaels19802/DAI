import depthai as dai
from depthai_sdk import OakCamera, TextPosition


with OakCamera() as oak:
    color = oak.create_camera('color')

    color.config_color_camera(scene_mode=dai.CameraControl.SceneMode.BARCODE,
                              awb_mode=dai.CameraControl.AutoWhiteBalanceMode.DAYLIGHT)

    nn = oak.create_nn('/home/michael/Dropbox/MichaelShared/BarcodeQRModel/barcode-and-qr-code-detection_2.json',
                       color, nn_type='yolo', spatial=True)

    nn.config_nn(conf_threshold=.1)
    visualizer = oak.visualize(nn, fps=True)

    visualizer.detections(label_position=TextPosition.TOP_LEFT,
                          fill_transparency=.3,
                          thickness=2,
                          hide_label=True,
                          ).text(auto_scale=False, font_scale=0.7, outline_color=(255, 255, 255))

    oak.start(blocking=True)
