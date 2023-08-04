import depthai as dai
from depthai_sdk import OakCamera, TextPosition


with OakCamera() as oak:
    color = oak.create_camera('color')

    color.config_color_camera(scene_mode=dai.CameraControl.SceneMode.BARCODE,
                              awb_mode=dai.CameraControl.AutoWhiteBalanceMode.DAYLIGHT)

    nn = oak.create_nn('/home/michael/Dropbox/MichaelShared/DepthAI/barcode-and-qr-code-detection_2_openvino_2022.1_6shave.blob',
                       color, nn_type='yolo')
    nn.config_yolo(num_classes=2,
                   coordinate_size=4,
                   anchors=[10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0,
                            59.0, 119.0, 116.0, 90.0, 156.0, 198.0, 373.0, 326.0],
                   masks={'side80': [0, 1, 2], 'side40': [3, 4, 5], 'side20': [6, 7, 8]},
                   iou_threshold=.5,
                   conf_threshold=.1)
    visualizer = oak.visualize(nn, fps=True)

    visualizer.detections(label_position=TextPosition.TOP_LEFT,
                          fill_transparency=.3,
                          thickness=3,
                          hide_label=True,
                          ).text(auto_scale=False, font_scale=1, outline_color=(255, 255, 255))

    oak.start(blocking=True)
