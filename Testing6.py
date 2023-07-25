import cv2

from depthai_sdk import OakCamera
from depthai_sdk.components.stereo_component import WLSLevel
from depthai_sdk.visualize.configs import StereoColor

with OakCamera() as oak:
    stereo = oak.create_stereo('800p', fps=10)
    stereo.config_postprocessing(colorize=StereoColor.RGBD, colormap=cv2.COLORMAP_DEEPGREEN)
    stereo.config_wls(wls_level=WLSLevel.MEDIUM)
    oak.visualize(stereo.out.depth, fps=True)
    oak.start(blocking=True)
