import depthai as dai
from depthai_sdk import OakCamera, TextPosition


with OakCamera() as oak:
    color = oak.create_camera('color')

    color.config_color_camera(scene_mode=dai.CameraControl.SceneMode.BARCODE,
                              awb_mode=dai.CameraControl.AutoWhiteBalanceMode.DAYLIGHT)

    model_config = {
        'source': 'roboflow',
        'model': 'barcode-and-qr-code-detection/2',
        'key': 'fAaLtouDRX3lJQdOgyJ1'
    }
    nn = oak.create_nn(model_config, color)

    nn.config_nn(conf_threshold=.1)
    visualizer = oak.visualize(nn, fps=True)

    visualizer.detections(label_position=TextPosition.MID,
                          fill_transparency=.4,
                          thickness=6,
                          ).text(auto_scale=False, font_thickness=1, outline_color=(255, 255, 255), background_color=(0, 0, 255))

    oak.start(blocking=True)
