from engine.Camera import Camera
from engine.FaceDetector import FaceDetector
from engine.BoxDrawer import BoxDrawer

import cv2
window_name = "Face recognition"
camera = Camera()
face_detector = FaceDetector()
box_drawer = BoxDrawer()

while camera.isOpened():
    is_returned, frame = camera.get_frame()

    if is_returned:

        results = face_detector.predict(frame)
        box_drawer.draw_boxes(results,frame)
        cv2.imshow(window_name,frame)

        k = cv2.waitKey(1)
        if k == 27 or cv2.getWindowProperty(window_name, 0) < 0:  # press ESC to exit
            camera.release()
            cv2.destroyAllWindows()
            break
    else:
        break
camera.release()