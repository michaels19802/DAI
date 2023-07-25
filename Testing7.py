import depthai as dai
from depthai_sdk import OakCamera


with OakCamera() as oak:
    color = oak.create_camera('color')
    nn = oak.create_nn('yolov6nr3_coco_640x352', color, spatial=True)
    nn.config_spatial(bb_scale_factor=0.5, lower_threshold=300, upper_threshold=10000,
                      calc_algo=dai.SpatialLocationCalculatorAlgorithm.AVERAGE)
    oak.visualize([nn.out.main], fps=True)
    oak.start(blocking=True)
