import cv2
import numpy as np

# Kalman filter class for smoothing
class KalmanFilter:
    """
    A Kalman filter for smoothing and predicting object trajectories.
    Tracks 4 state variables: (x, y, vx, vy) and 2 measurement variables: (x, y).
    """

    def __init__(self):
        # Create a Kalman Filter object with 4 state variables
        # and 2 measurements variables using cv2
        self.kf = cv2.KalmanFilter(4, 2)

        # Preserve the measurements
        self.last_measured_center = None
        self.last_norm_ratio = None

        # Set up the measurement matrix 'H' as a 2x4 matrix that
        # maps 'x' and 'y' coordinates to our 4-dimensional state vector
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], np.float32)
        
        # The transition matrix 'A' defines how our state vectors evolve from
        # time step 't' to 't+1' based on a simple linear motion model
        # First two rows map position estimates onto future position estimates
        # Last two rows mantain unchanged predictions about velocities
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], np.float32)
        
        # The process noise covariance matrix 'Q' represents the uncertainty in our
        # motion model and affects how the Kalman filter predicts the next state
        # It is set as a diagonal matrix scaled by 0.03, meaning that we're adding
        # small errors to each of our 4 variables
        self.kf.processNoiseCov = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], np.float32) * 0.015
        
        # The measurement noise covariance matrix 'R' contains the variance of the
        # measurements
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0

        # Error covariance matrix 'P'
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)

        # State post initialization
        self.kf.statePost = np.zeros((4, 1), dtype=np.float32)

        self.initialized = False

    def update(self, measurement):
        """
        Updates the Kalman filter with a new measurement and predicts the next state.
        Args:
            measurement (tuple): (x, y) coordinates of the measured object.
        Returns:
            np.array: Smoothed (x, y) coordinates.
        """
        measurement = np.array(measurement, dtype=np.float32).reshape(2, 1)
        if not self.initialized:
            # Initialize the state with the measurement
            self.kf.statePost = np.vstack((measurement, np.zeros((2, 1), dtype=np.float32)))
            self.initialized = True
            return measurement.flatten()
        else:
            self.kf.correct(measurement)
            prediction = self.kf.predict()
            return prediction[:2].flatten()