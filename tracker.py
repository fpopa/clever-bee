# Import python libraries
import numpy as np
from scipy.optimize import linear_sum_assignment

from track import Track
min_tracker_score_creation = 20

class Tracker(object):
    def __init__(self, dist_thresh, max_frames_to_skip, max_trace_length,
                 trackIdCount):
        """Initialize variable used by Tracker class
        Args:
            dist_thresh: distance threshold. When exceeds the threshold,
                         track will be deleted and new track is created
            max_frames_to_skip: maximum allowed frames to be skipped for
                                the track object undetected
            max_trace_lenght: trace path history length
            trackIdCount: identification of each track object
        Return:
            None
        """
        self.dist_thresh = dist_thresh
        self.max_frames_to_skip = max_frames_to_skip
        self.max_trace_length = max_trace_length
        self.tracks = []
        self.trackIdCount = trackIdCount

    def Update(self, detectionsConfidence):
        detections = []
        for i in range(len(detectionsConfidence)):
            detections.append(np.round(np.array([[detectionsConfidence[i][1]], [detectionsConfidence[i][2]]])))
        """Update tracks vector using following steps:
            - Create tracks if no tracks vector found
            - Calculate cost using sum of square distance
              between predicted vs detected centroids
            - Using Hungarian Algorithm assign the correct
              detected measurements to predicted tracks
            - If tracks are not detected for long time, remove them
            - if no track and un_assigned detects
            - Start new track for highest confidence detection
            - Update KalmanFilter state, lastResults and tracks trace
        Args:
            detections: detected centroids of object to be tracked
        Return:
            None
        """

        # Create tracks if no tracks vector found
        if (len(self.tracks) == 0):
            if (len(detections) > 0):
                if (detectionsConfidence[0][0] > min_tracker_score_creation):
                    track = Track(detections[0], self.trackIdCount)

                    self.trackIdCount += 1
                    self.tracks.append(track)

        # Calculate cost using sum of square distance between predicted vs detected centroids
        N = len(self.tracks)
        M = len(detections)
        cost = np.zeros(shape=(N, M))   # Cost matrix
        for i in range(len(self.tracks)):
            self.tracks[i].age += 1
            for j in range(len(detections)):
                pred_point = [self.tracks[i].prediction[0], self.tracks[i].prediction[1]]

                # print (pred_point)
                # print (detections[j])

                diff = pred_point - detections[j]
                distance = np.sqrt(diff[0][0]*diff[0][0] +
                                   diff[1][0]*diff[1][0])
                cost[i][j] = distance

        # Let's average the squared ERROR
        cost = (0.5) * cost
        # Using Hungarian Algorithm assign the
        # correct detected measurements to predicted tracks
        assignment = []
        for _ in range(N):
            assignment.append(-1)
        row_ind, col_ind = linear_sum_assignment(cost)
        for i in range(len(row_ind)):
            assignment[row_ind[i]] = col_ind[i]
            # print (str(row_ind[i]) + ' -> ' + str(col_ind[i]) + ' = ' + str(cost[row_ind[i]][col_ind[i]]))

        # Identify tracks with no assignment, if any
        un_assigned_tracks = []
        for i in range(len(assignment)):
            if (assignment[i] != -1):
                # check for cost distance threshold.
                # If cost is very high then un_assign (delete) the track
                if (cost[i][assignment[i]] > self.dist_thresh):
                    assignment[i] = -1
                    un_assigned_tracks.append(i)
                pass
            else:
                self.tracks[i].skipped_frames += 1

        # print ('--------------')
        # print (cost)
        # print (assignment)
        # print ('--------------')

        # If tracks are not detected for a long time, remove them
        del_tracks = []
        for i in range(len(self.tracks)):
            # print (self.tracks[i].skipped_frames)
            if (self.tracks[i].skipped_frames > self.max_frames_to_skip):
                del_tracks.append(i)
        if len(del_tracks) > 0:  # only when skipped frame exceeds max
            for id in del_tracks:
                if id < len(self.tracks):
                    del self.tracks[id]
                    del assignment[id]
                else:
                    print("ERROR: id is greater than length of tracks")

        # This should be enabled for multi tracking
            # # Now look for un_assigned detects
            # un_assigned_detects = []
            # for i in range(len(detections)):
            #         if i not in assignment:
            #             un_assigned_detects.append(i)

            # # Start new tracks
            # if(len(un_assigned_detects) != 0):
            #     for i in range(len(un_assigned_detects)):
            #         if (detectionsConfidence[un_assigned_detects[i]][0] > min_tracker_score_creation):
            #             track = Track(detections[un_assigned_detects[i]], self.trackIdCount)

            #             self.trackIdCount += 1
            #             self.tracks.append(track)

        # Update KalmanFilter state, lastResults and tracks trace
        for i in range(len(assignment)):
            self.tracks[i].prediction = self.tracks[i].KF.predict()

            self.tracks[i].last_detection_assigment = None
            if(assignment[i] != -1):
                self.tracks[i].skipped_frames = 0
                self.tracks[i].prediction = self.tracks[i].KF.correct(detections[assignment[i]])
                self.tracks[i].last_detection_assigment = assignment[i]

            # cv2.circle(frame, tuple(np.round(self.tracks[i].prediction)), 4, (255, 255, 255), 2)

            if(len(self.tracks[i].trace) > self.max_trace_length):
                for j in range(len(self.tracks[i].trace) -
                               self.max_trace_length):
                    del self.tracks[i].trace[j]

            # self.tracks[i].trace.append(self.tracks[i].prediction)
            self.tracks[i].trace.append([
                int(self.tracks[i].prediction[0]),
                int(self.tracks[i].prediction[1])
            ])

            # print (len(self.tracks[i].trace))
            # print (self.tracks[i].trace)

            self.tracks[i].KF.lastResult = self.tracks[i].prediction
