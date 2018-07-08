# Import python libraries
import cv2, numpy as np


class Kalman(object):
    """Kalman Filter class keeps track of the estimated state of
    the system and the variance or uncertainty of the estimate.
    Predict and Correct methods implement the functionality
    Reference: https://en.wikipedia.org/wiki/Kalman_filter
    Attributes: None
    """

    def __init__(self, initial_point):
        self.kalman = cv2.KalmanFilter(4,2)
        self.kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
        self.kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
        self.kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * 0.03

        self.kalman.statePost = np.array([[initial_point[0]], [initial_point[1]], [0], [0]], dtype=np.float32)

    def predict(self):
        val = tuple(self.kalman.predict())
        return val;

    def correct(self, mp):
        val = self.kalman.correct(np.array(mp, dtype=np.float32))
        return val
