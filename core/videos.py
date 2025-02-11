import os
import cv2
import dlib
import imutils
import numpy as np
import tensorflow as tf

from typing import List
from utils.kalman_filter import KalmanFilter

class Video(object):
    def __init__(self, roi_width=100, roi_height=50, horizontal_pad=0.19, face_predictor_path=None):
        if not face_predictor_path or not os.path.exists(os.path.expanduser(face_predictor_path)):
            raise AttributeError('Video must have a face predictor. Check the path provided.')
        self.face_predictor_path = os.path.expanduser(face_predictor_path)
        
        self.roi_width = roi_width
        self.roi_height = roi_height
        self.horizontal_pad = horizontal_pad

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.face_predictor_path)
        self.kalman = KalmanFilter()
    
    # Add padding to resize to 100x50 while keeping aspect ratio
    def resize_with_padding(self, image):
        # Check if the image is valid (non-empty)
        if image is None or image.shape[0] == 0 or image.shape[1] == 0:
            return None  # Return None if the image is invalid
        
        h, w = image.shape[:2]

        # Calculate aspect ratio
        aspect_ratio = w / h

        # Determine the new size while maintaining the aspect ratio
        if aspect_ratio > 1: # Landscape (wider than tall)
            new_w = self.roi_width
            new_h = int(self.roi_width / aspect_ratio)
        else: # Portrait (taller than wide or square)
            new_h = self.roi_height
            new_w = int(self.roi_height * aspect_ratio)

        # Resize the image to fit within the target size while keeping the aspect ratio
        resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Add padding to make the image the target size (100x50 e.g.)
        pad_t = (self.roi_height - new_h) // 2
        pad_b = self.roi_height - new_h - pad_t
        pad_l = (self.roi_width - new_w) // 2
        pad_r = self.roi_width - new_w - pad_l

        pad_t = max(0, pad_t)
        pad_b = max(0, pad_b)
        pad_l = max(0, pad_l)
        pad_r = max(0, pad_r)

        # Add the padding to the resized image
        padded_img = cv2.copyMakeBorder(resized_img, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        
        # Force the final image to be exactly target size (width x height)
        final_img = cv2.resize(padded_img, (self.roi_width, self.roi_height), interpolation=cv2.INTER_AREA)
        return final_img

    def extract_mouth_region(self, image):
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.detector(gray, 1)
        if len(faces) > 0:
            shape = self.predictor(gray, faces[0])
        else:
            shape = None

        # If no face is detected, use the kalman filter's prediction
        normalize_ratio = None
        mouth_centroid_norm = None
        if shape is None:
            if self.kalman is not None and self.kalman.last_norm_ratio is not None and self.kalman.last_measured_center is not None:
                mouth_centroid_norm = self.kalman.update(self.kalman.last_measured_center)
                normalize_ratio = self.kalman.last_norm_ratio
            else:
                print("Kalman filter prediction unavailable...")
                fallback_image = imutils.resize(image, self.roi_width, self.roi_height)
                return self.resize_with_padding(fallback_image)
        else:
            # Extract mouth landmarks from the detected shape
            mouth_points = []
            i = -1
            for part in shape.parts():
                i += 1
                if i < 48: # Only take mouth region
                    continue
                mouth_points.append((part.x, part.y))
            np_mouth_points = np.array(mouth_points)

            # Compute mouth centroid
            mouth_centroid = np.mean(np_mouth_points[:, -2:], axis=0)

            if normalize_ratio is None:
                mouth_left = np.min(np_mouth_points[:, :-1]) * (1.0 - self.horizontal_pad)
                mouth_right = np.max(np_mouth_points[:, :-1]) * (1.0 + self.horizontal_pad)

                normalize_ratio = self.roi_width / float(mouth_right - mouth_left)
                self.kalman.last_norm_ratio = normalize_ratio
            
            mouth_centroid_norm = mouth_centroid * normalize_ratio
            # Apply the kalman filter
            if self.kalman:
                mouth_centroid_norm = self.kalman.update(mouth_centroid_norm)
            self.kalman.last_measured_center = mouth_centroid_norm

        # Assume last_norm_ratio is valid if detection failed
        if self.kalman.last_norm_ratio is None:
            self.kalman.last_norm_ratio = 1.0 # fallback

        new_img_shape = (int(image.shape[0] * normalize_ratio), int(image.shape[1] * normalize_ratio))
        resized_img = cv2.resize(image, (new_img_shape[1], new_img_shape[0]), interpolation=cv2.INTER_AREA)
        
        mouth_l = int(mouth_centroid_norm[0] - self.roi_width / 2)
        mouth_r = int(mouth_centroid_norm[0] + self.roi_width / 2)
        mouth_t = int(mouth_centroid_norm[1] - self.roi_height / 2)
        mouth_b = int(mouth_centroid_norm[1] + self.roi_height / 2)

        try:
            mouth_roi = resized_img[mouth_t:mouth_b, mouth_l:mouth_r]
            if mouth_roi.shape[0] == 0 or mouth_roi.shape[1] == 0:
                raise Exception("Mouth ROI not detected in frame...")
        except Exception as e:
            # Use fallback image
            print(f"Using fallback image {e}")
            fallback_image = imutils.resize(image, self.roi_width, self.roi_height)
            return self.resize_with_padding(fallback_image)

        mouth_roi = self.resize_with_padding(mouth_roi)
        
        return mouth_roi

    # Function to load the video and preprocess it
    def load_video(self, path:str) -> List[float]:
        cap = cv2.VideoCapture(path)

        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            print(f"Warning: No frames found in video: {path}")
            cap.release()
            return tf.constant([])
        
        for _ in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                print("Cannot open video. Exiting...")
                break
            # Extract the mouth region of the frame
            mouth_roi = self.extract_mouth_region(frame)
            
            if mouth_roi is None:
                print("Warning: Mouth region extraction failed for the frame. Skipping...")
                continue
            frames.append(mouth_roi)
        cap.release()

        if not frames:
            print(f"Warning: No valid frames were extracted from the video: {path}")
            return tf.constant([])
        
        mean = tf.math.reduce_mean(tf.convert_to_tensor(frames, dtype=tf.float32))
        std = tf.math.reduce_std(tf.cast(frames, tf.float32))

        epsilon = 1e-8
        return tf.cast((tf.convert_to_tensor(frames, dtype=tf.float32) - mean), tf.float32) / (std + epsilon)