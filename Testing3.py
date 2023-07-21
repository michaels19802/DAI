import depthai_sdk as dsdk


with dsdk.OakCamera(usb_speed='3') as oak:
    color = oak.create_camera('color', resolution='1080p', fps=20)
    stereo = oak.create_stereo(resolution='800p', fps=20)
    oak.visualize([color, stereo])
    oak.start(blocking=True)
