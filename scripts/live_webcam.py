#!/usr/bin/env python3
# --- Add project root to sys.path ---
import sys
import os
import cv2

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
# ------------------------------------

from core.videos import Video

# "~/.dlib/shape_predictor_68_face_landmarks.dat"

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot Open Camera!")
        exit()

    # Adjust the frame size
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    video = Video(face_predictor_path="~/.dlib/shape_predictor_68_face_landmarks.dat")
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # if frame is read corrrectly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # Operations on the frame come here
        mouth_roi = video.extract_mouth_region(frame)

        # Display the mouth ROI
        if mouth_roi is not None:
            cv2.imshow("mouth roi", mouth_roi)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()