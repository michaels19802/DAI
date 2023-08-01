import depthai as dai
from depthai_sdk import OakCamera, TextPosition, BboxStyle


with OakCamera() as oak:
    color = oak.create_camera('color')
    nn = oak.create_nn('yolov6nr3_coco_640x352', color, spatial=True)
    nn.config_spatial(bb_scale_factor=.5, lower_threshold=300, upper_threshold=10000,
                      calc_algo=dai.SpatialLocationCalculatorAlgorithm.AVERAGE)
    visualizer = oak.visualize(nn.out.main, fps=True)

    visualizer.detections(bbox_style=BboxStyle.RECTANGLE,
                          label_position=TextPosition.TOP_LEFT,
                          fill_transparency=.5,
                          thickness=2
                          ).text(auto_scale=False, font_thickness=1, outline_color=(255, 255, 255))

    oak.start(blocking=True)
