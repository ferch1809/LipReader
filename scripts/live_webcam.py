#!/usr/bin/env python3
# --- Add project root to sys.path ---
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
# ------------------------------------

import cv2
from preprocessing.mouth_extraction import extract_mouth_region
from utils.kalman_filter import KalmanFilter

def main():
    cap = cv2.VideoCapture(0)
    kalman_filter = KalmanFilter()

    if not cap.isOpened():
        print("Cannot Open Camera!")
        exit()

    # Adjust the frame size
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # if frame is read corrrectly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # Operations on the frame come here
        mouth_roi = extract_mouth_region(frame, kalman=kalman_filter)

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