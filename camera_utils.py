import cv2

# Global config (single source of truth)
WINDOW_NAME = "EduVision Camera"
WINDOW_WIDTH = 1100
WINDOW_HEIGHT = 750

CAM_WIDTH = 960
CAM_HEIGHT = 720


def setup_camera(cam_index=0):
    cam = cv2.VideoCapture(cam_index)

    if not cam.isOpened():
        cam = cv2.VideoCapture(0)

    cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    # Quick warmup — 5 frames is enough for macOS AVFoundation
    for _ in range(5):
        cam.read()

    return cam


def setup_window(title=WINDOW_NAME):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, WINDOW_WIDTH, WINDOW_HEIGHT)
    return title