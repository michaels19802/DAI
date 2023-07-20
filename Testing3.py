import depthai_sdk as dsdk

dsdk.set_logging_level(10)

with dsdk.OakCamera(usb_speed='2') as oak:
    color = oak.create_camera('color', resolution='1080p')
    stereo = oak.create_stereo(resolution='800p')
    oak.visualize([color, stereo])
    oak.start(blocking=True)
