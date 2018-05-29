'''
    File name         : tracker.py
    File Description  : Tracker Using Kalman Filter & Hungarian Algorithm
    Author            : Srini Ananthakrishnan
    Date created      : 07/14/2017
    Date last modified: 07/16/2017
    Python Version    : 2.7
'''

# Import python libraries
import numpy as np
from kalman_filter import KalmanFilter,KalmanFilter_cv
from common import dprint
from scipy.optimize import linear_sum_assignment
from confg import config


class Track(object):
    """Track class for every object to be tracked
    Attributes:
        None
    """

    def __init__(self, prediction, trackIdCount):
        """Initialize variables used by Track class
        Args:
            prediction: predicted centroids of object to be tracked
            trackIdCount: identification of each track object
        Return:
            None
        """
        self.track_id = trackIdCount  # identification of each track object
        if config.tracker_cv:
            self.KF = KalmanFilter_cv()
        else:
            self.KF = KalmanFilter()  # KF instance to track this object
        self.prediction = np.asarray(prediction)  # predicted centroids (x,y)
        self.skipped_frames = 0  # number of frames skipped undetected
        self.trace = []  # trace path


class Tracker(object):
    """Tracker class that updates track vectors of object tracked
    Attributes:
        None
    """

    def __init__(self, dist_thresh, max_frames_to_skip, max_trace_length,
                 trackIdCount,w,h):
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
        self.predict_center = []
        self.assign_out = []
        self.width = w
        self.height = h

    def Update(self, detections,rectw_h,key_frame):
        """Update tracks vector using following steps:
            - Create tracks if no tracks vector found
            - Calculate cost using sum of square distance
              between predicted vs detected centroids
            - Using Hungarian Algorithm assign the correct
              detected measurements to predicted tracks
              https://en.wikipedia.org/wiki/Hungarian_algorithm
            - Identify tracks with no assignment, if any
            - If tracks are not detected for long time, remove them
            - Now look for un_assigned detects
            - Start new tracks
            - Update KalmanFilter state, lastResults and tracks trace
        Args:
            detections: detected centroids of object to be tracked
        Return:
            None
        """
        #initializer
        self.predict_center = []
        rect_w,rect_h = rectw_h[:,0],rectw_h[:,1]
        #detect_box_num = len(rectw_h)
        # Create tracks if no tracks vector found
        if config.Debug:
            print("-----------------traker begin")
        if (len(self.tracks) == 0):
            for i in range(len(detections)):
                track = Track(detections[i], self.trackIdCount)
                if config.Debug:
                    print("tracker init,ID: ",self.trackIdCount)
                self.trackIdCount += 1
                self.tracks.append(track)
        # Calculate cost using sum of square distance between
        # predicted vs detected centroids
        N = len(self.tracks)
        M = len(detections)
        if config.Debug:
            #print("-----------------traker begin")
            print("traker and detect ",N,M)
        cost = np.zeros(shape=(N, M))   # Cost matrix
        for i in range(len(self.tracks)):
            for j in range(len(detections)):
                try:
                    diff = self.tracks[i].prediction - detections[j]
                    distance = np.sqrt(diff[0][0]*diff[0][0] +
                                       diff[1][0]*diff[1][0])
                    cost[i][j] = distance
                except:
                    pass
        # Let's average the squared ERROR
        cost = (0.5) * cost
        if key_frame:
            # Using Hungarian Algorithm assign the correct detected measurements
            # to predicted tracks
            self.assignment = []
            for _ in range(N):
                self.assignment.append(-1)
            row_ind, col_ind = linear_sum_assignment(cost)
            for i in range(len(row_ind)):
                self.assignment[row_ind[i]] = col_ind[i]
            if config.Debug :
                print("row_id ",row_ind)
                print("col_id ",col_ind)
                print("assignment ",self.assignment)
        # Identify tracks with no assignment, if any
        un_assigned_tracks = []
        del_key_tracks = []
        for i in range(len(self.assignment)):
            if (self.assignment[i] != -1):
                # check for cost distance threshold.
                # If cost is very high then un_assign (delete) the track
                '''
                left =  1 if self.tracks[i].prediction[0][0] - rect_w[i]/2 <0 else 0
                right = 1 if self.tracks[i].prediction[0][0] + rect_w[i]/2 >= self.width else 0
                top = 1 if self.tracks[i].prediction[1][0] - rect_h[i]/2 <0 else 0
                bottom = 1 if self.tracks[i].prediction[1][0] + rect_h[i]/2 >= self.height else 0
                '''
                #if (cost[i][assignment[i]] > self.dist_thresh or left or right or top or bottom):
                if (cost[i][self.assignment[i]] > self.dist_thresh ):
                    #print("-------------,cost[i][assignment[i]],left,right,top,bottom",cost[i][assignment[i]],left,right,top,bottom)
                    print("-------------,cost[i][assignment[i]]",cost[i][self.assignment[i]])
                    self.assignment[i] = -1
                    un_assigned_tracks.append(i)
                    self.tracks[i].skipped_frames += 1
                    '''
                    if key_frame:
                        del assignment[i]
                        del self.tracks[i]
                    '''
                pass
            elif key_frame:
                '''
                del assignment[i]
                del self.tracks[i]
                if config.Debug:
                    print("after key_frame del assignment ", assignment)
                '''
                del_key_tracks.append(i)
                self.tracks[i].skipped_frames += 1
            else:
                self.tracks[i].skipped_frames += 1
                if config.Debug:
                    print("-1 key_frame",key_frame)
        '''
        if len(un_assigned_tracks) and len(del_key_tracks):
            sort_tmp = np.array([un_assigned_tracks[:],del_key_tracks[:]])
            sort_tmp = np.sort(sort_tmp,axis=None)[::-1]
            for j in sort_tmp:
                del assignment[j]
                del self.tracks[j]
                if config.Debug:
                    print("key and always del assignment ", assignment)
        elif len(un_assigned_tracks):
            sort_tmp = np.sort(un_assigned_tracks)[::-1]
            for j in sort_tmp:
                del assignment[j]
                del self.tracks[j]
                if config.Debug:
                    print("always del assignment ", assignment)
        elif len(del_key_tracks):
            sort_tmp = np.sort(del_key_tracks)[::-1]
            for j in sort_tmp:
                del assignment[j]
                del self.tracks[j]
                if config.Debug:
                    print("key  del assignment ", assignment)
        '''
        # If tracks are not detected for long time, remove them
        self.del_tracks = []
        for i in range(len(self.tracks)):
            if (self.tracks[i].skipped_frames > self.max_frames_to_skip):
                self.del_tracks.append(i)
        if len(self.del_tracks) > 0:  # only when skipped frame exceeds max
            sort_tmp = np.sort(self.del_tracks)[::-1]
            for id in sort_tmp:
                if id < len(self.tracks):
                    del self.tracks[id]
                    del self.assignment[id]
                    print("-----------del a face")
                else:
                    dprint("ERROR: id is greater than length of tracks")
        # Now look for un_assigned detects
        un_assigned_detects = []
        for i in range(len(detections)):
                if i not in self.assignment and key_frame :
                    if config.Debug:
                        print(" new track will: tracker_num,det_num and i ,key_frame",len(self.tracks),len(detections),i,key_frame,self.assignment)
                    un_assigned_detects.append(i)
        # Start new tracks
        if(len(un_assigned_detects) != 0):
            print("start new track")
            for i in range(len(un_assigned_detects)):
                track = Track(detections[un_assigned_detects[i]],self.trackIdCount)
                if config.Debug:
                    print("assign Id: ",self.trackIdCount)
                self.trackIdCount += 1
                self.tracks.append(track)
                if key_frame:
                    self.assignment.append(un_assigned_detects[i])
        # Update KalmanFilter state, lastResults and tracks trace
        #for i in range(len(assignment)):
        for i in range(len(detections)):
            self.tracks[i].KF.predict()
            #if(assignment[i] != -1):
            if i in self.assignment:
                self.tracks[i].skipped_frames = 0
                if config.tracker_cv :
                    pos = self.tracks[i].KF.correct(detections[self.assignment[i]])
                    c_x = pos[0]
                    c_y = pos[1]
                    w = detections[self.assignment[i]][2]
                    h = detections[self.assignment[i]][3]
                    self.tracks[i].prediction = [c_x,c_y,w,h]
                else:
                    self.tracks[i].prediction = self.tracks[i].KF.correct(detections[self.assignment[i]], 1)
            else:
                self.tracks[i].prediction = self.tracks[i].KF.correct(detections[i],1)
                #self.tracks[i].prediction = self.tracks[i].prediction
            if(len(self.tracks[i].trace) > self.max_trace_length):
                for j in range(len(self.tracks[i].trace) -self.max_trace_length):
                    del self.tracks[i].trace[j]
            if config.tracker_cv:
                self.tracks[i].trace.append(self.tracks[i].prediction[:2])
            else:
                self.tracks[i].trace.append(self.tracks[i].prediction)
            if  config.tracker_cv:
                self.predict_center.append(self.tracks[i].prediction)
            else:
                self.tracks[i].KF.lastResult = self.tracks[i].prediction
                x = self.tracks[i].prediction[0][0]
                y = self.tracks[i].prediction[1][0]
                self.predict_center.append(np.array([[x],[y]],dtype=np.int))
        self.assign_out = self.assignment
        if config.Debug:
            print("---------------trackor over ",len(self.tracks))
