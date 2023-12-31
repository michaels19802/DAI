from depthai_sdk import OakCamera


with OakCamera() as oak:
    cams = oak.create_all_cameras()
    oak.visualize(cams, scale=.4, fps=True)
    oak.start(blocking=True)
