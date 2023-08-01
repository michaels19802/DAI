import depthai_sdk as dsdk


with dsdk.OakCamera(usb_speed='3') as oak:
    color = oak.create_camera('color', resolution='1080p')
    stereo = oak.create_stereo(resolution='800p')
    oak.visualize([color, stereo], fps=True)
    oak.start(blocking=True)
