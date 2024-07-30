from ultralytics import YOLO
import cv2


def image_detection(path_x):
    image_capture = path_x
    model = YOLO('Plant_detector.pt')
    results = model(image_capture, show=True, save=True)
    cv2.waitKey(0)
    return results
