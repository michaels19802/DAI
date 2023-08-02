import blobconverter
from depthai_sdk import OakCamera


with OakCamera() as oak:
    color = oak.create_camera('color', '1080p', fps=15)
    nn_path = blobconverter.from_zoo(name='qr_code_detection_384x384', zoo_type='depthai')
    nn = oak.create_nn(nn_path, color, 'MobileNet')

    nn.config_nn(conf_threshold=0.01)
    visualizer = oak.visualize(nn.out.main, fps=True)
    visualizer.detections(hide_label=True, color=(0, 0, 255), thickness=3, fill_transparency=.5).text(auto_scale=False)
    oak.start(blocking=True)
