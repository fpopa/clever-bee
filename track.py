import numpy as np
from kalman import Kalman

class Track(object):
    """Track class for every object to be tracked
    Attributes:
        None
    """

    def __init__(self, initial_point, trackIdCount):
        """Initialize variables used by Track class
        Args:
            prediction: predicted centroids of clusters to be tracked
            trackIdCount: identification of each track object
        Return:
            None
        """
        self.track_id = trackIdCount  # identification of each track object
        self.KF = Kalman(initial_point)  # KF instance to track this object
        self.prediction = np.asarray(initial_point)  # predicted centroids (x,y)
        self.skipped_frames = 0  # number of frames skipped undetected
        self.last_detection_assigment = 0  # number of frames skipped undetected
        self.age = 0 # the number of frames this tracker has lived for
        self.trace = []  # trace path
