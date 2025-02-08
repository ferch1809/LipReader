import os
import cv2
import dlib

import numpy as np
import imutils

# Fixed output size
MOUTH_WIDTH = 100
MOUTH_HEIGHT = 50
HORIZONTAL_PAD = 0.19

# Initialize dlib's face detector (HOG-based)
detector = dlib.get_frontal_face_detector()
# Create the facial landmark predictor (update path to pre-trained model as necessary)
predictor = dlib.shape_predictor(
    os.path.expanduser("~/.dlib/shape_predictor_68_face_landmarks.dat")
)

# Function to crop the points of the cropped ROI
def get_crop_points(centroid):
    mouth_l = int(centroid[0] - MOUTH_WIDTH / 2)
    mouth_r = int(centroid[0] + MOUTH_WIDTH / 2)
    mouth_t = int(centroid[1] - MOUTH_HEIGHT / 2)
    mouth_b = int(centroid[1] + MOUTH_HEIGHT / 2)

    return mouth_l, mouth_r, mouth_t, mouth_b

# Add padding to resize to 100x50 while keeping aspect ratio
def resize_with_padding(image, target_size=(MOUTH_WIDTH, MOUTH_HEIGHT)):
    target_w, target_h = target_size

    # Check if the image is valid (non-empty)
    if image is None or image.shape[0] == 0 or image.shape[1] == 0:
        return None  # Return None if the image is invalid
    
    h, w = image.shape[:2]

    # Calculate aspect ratio
    aspect_ratio = w / h

    # Determine the new size while maintaining the aspect ratio
    if aspect_ratio > 1: # Landscape (wider than tall)
        new_w = target_w
        new_h = int(target_w / aspect_ratio)
    else: # Portrait (taller than wide or square)
        new_h = target_h
        new_w = int(target_h * aspect_ratio)

    # Resize the image to fit within the target size while keeping the aspect ratio
    resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Add padding to make the image the target size (100x50 e.g.)
    pad_t = (target_h - new_h) // 2
    pad_b = target_h - new_h - pad_t
    pad_l = (target_w - new_w) // 2
    pad_r = target_w - new_w - pad_l

    pad_t = max(0, pad_t)
    pad_b = max(0, pad_b)
    pad_l = max(0, pad_l)
    pad_r = max(0, pad_r)

    # Add the padding to the resized image
    padded_img = cv2.copyMakeBorder(resized_img, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    
    # Force the final image to be exactly target size (width x height)
    final_img = cv2.resize(padded_img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return final_img

def extract_mouth_region(image, kalman=None):
    """
    Extracts and processes the mouth region from an image using dlib landmarks.
    Optionally smooths the detected mouth center using a Kalman filter.

    Args:
        image (np.array): Input image in BGR format.
        kalman (KalmanFilter, optional): Kalman filter instance for smoothing. Defaults to None.

    Returns:
        tuple: Processed image of mouth ROI (100x50)
    """
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector(gray, 1)
    if len(faces) > 0:
        shape = predictor(gray, faces[0])
    else:
        print("No faces detected")
        shape = None

    # If no face is detected, use the kalman filter's prediction
    normalize_ratio = None
    mouth_centroid_norm = None
    if shape is None:
        if kalman is not None and kalman.last_norm_ratio is not None and kalman.last_measured_center is not None:
            mouth_centroid_norm = kalman.update(kalman.last_measured_center)
            normalize_ratio = kalman.last_norm_ratio
        else:
            print("Kalman filter prediction unavailable...")
            fallback_image = imutils.resize(image, MOUTH_WIDTH, MOUTH_HEIGHT)
            return resize_with_padding(fallback_image)
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
            mouth_left = np.min(np_mouth_points[:, :-1]) * (1.0 - HORIZONTAL_PAD)
            mouth_right = np.max(np_mouth_points[:, :-1]) * (1.0 + HORIZONTAL_PAD)

            normalize_ratio = MOUTH_WIDTH / float(mouth_right - mouth_left)
            kalman.last_norm_ratio = normalize_ratio
        
        mouth_centroid_norm = mouth_centroid * normalize_ratio
        # Apply the kalman filter
        if kalman:
            mouth_centroid_norm = kalman.update(mouth_centroid_norm)
        kalman.last_measured_center = mouth_centroid_norm

    # Assume last_norm_ratio is valid if detection failed
    if kalman.last_norm_ratio is None:
        kalman.last_norm_ratio = 1.0 # fallback

    new_img_shape = (int(image.shape[0] * normalize_ratio), int(image.shape[1] * normalize_ratio))
    resized_img = cv2.resize(image, (new_img_shape[1], new_img_shape[0]), interpolation=cv2.INTER_AREA)
    
    crop_points = get_crop_points(mouth_centroid_norm)

    try:
        mouth_roi = resized_img[crop_points[2]:crop_points[3],crop_points[0]:crop_points[1]]
        if mouth_roi.shape[0] == 0 or mouth_roi.shape[1] == 0:
            raise ValueError("Mouth ROI is empty")
    except Exception as e:
        print(f"Warning: Mouth region extraction failed: {str(e)}")
        # Use fallback image
        fallback_image = imutils.resize(image, MOUTH_WIDTH, MOUTH_HEIGHT)
        return resize_with_padding(fallback_image)

    mouth_roi = resize_with_padding(mouth_roi)
    # print(f"width:{mouth_roi.shape[0]} height:{mouth_roi.shape[1]}")
    
    return mouth_roi