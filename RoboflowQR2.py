from depthai_sdk import OakCamera, TextPosition, BboxStyle


with OakCamera() as oak:
    color = oak.create_camera('color')
    model_config = {
        'source': 'roboflow',
        'model': 'barcode-and-qr-code-detection/2',
        'key': 'fAaLtouDRX3lJQdOgyJ1'
    }
    nn = oak.create_nn(model_config, color)

    nn.config_nn(conf_threshold=.1)
    visualizer = oak.visualize(nn, fps=True)

    visualizer.detections(bbox_style=BboxStyle.CORNERS,
                          label_position=TextPosition.TOP_LEFT,
                          fill_transparency=.5,
                          thickness=2,
                          ).text(auto_scale=False, font_thickness=1, outline_color=(255, 255, 255), background_color=(0, 0, 255))

    oak.start(blocking=True)
