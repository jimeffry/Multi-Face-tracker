'''
    File name         : kalman_filter.py
    File Description  : Kalman Filter Algorithm Implementation
    Author            : Srini Ananthakrishnan
    Date created      : 07/14/2017
    Date last modified: 07/16/2017
    Python Version    : 2.7
'''

# Import python libraries
import numpy as np
import cv2

class KalmanFilter(object):
    """Kalman Filter class keeps track of the estimated state of
    the system and the variance or uncertainty of the estimate.
    Predict and Correct methods implement the functionality
    Reference: https://en.wikipedia.org/wiki/Kalman_filter
    Attributes: None
    """

    def __init__(self):
        """Initialize variable used by Kalman Filter class
        Args:
            None
        Return:
            None
        """
        self.dt = 0.005  # delta time

        self.A = np.array([[1, 0], [0, 1]])  # matrix in observation equations
        self.u = np.zeros((2, 1))  # previous state vector

        # (x,y) tracking object center
        self.b = np.array([[0], [255]])  # vector of observations

        self.P = np.diag((3.0, 3.0))  # covariance matrix
        self.F = np.array([[1.0, self.dt], [0.0, 1.0]])  # state transition mat

        self.Q = np.eye(self.u.shape[0])  # process noise matrix
        self.R = np.eye(self.b.shape[0])  # observation noise matrix
        self.lastResult = np.array([[0], [255]])

    def predict(self):
        """Predict state vector u and variance of uncertainty P (covariance).
            where,
            u: previous state vector
            P: previous covariance matrix
            F: state transition matrix
            Q: process noise matrix
        Equations:
            u'_{k|k-1} = Fu'_{k-1|k-1}
            P_{k|k-1} = FP_{k-1|k-1} F.T + Q
            where,
                F.T is F transpose
        Args:
            None
        Return:
            vector of predicted state estimate
        """
        # Predicted state estimate
        self.u = np.round(np.dot(self.F, self.u))
        # Predicted estimate covariance
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        self.lastResult = self.u  # same last predicted result
        return self.u

    def correct(self, b, flag):
        """Correct or update state vector u and variance of uncertainty P (covariance).
        where,
        u: predicted state vector u
        A: matrix in observation equations
        b: vector of observations
        P: predicted covariance matrix
        Q: process noise matrix
        R: observation noise matrix
        Equations:
            C = AP_{k|k-1} A.T + R
            K_{k} = P_{k|k-1} A.T(C.Inv)
            u'_{k|k} = u'_{k|k-1} + K_{k}(b_{k} - Au'_{k|k-1})
            P_{k|k} = P_{k|k-1} - K_{k}(CK.T)
            where,
                A.T is A transpose
                C.Inv is C inverse
        Args:
            b: vector of observations
            flag: if "true" prediction result will be updated else detection
        Return:
            predicted state vector u
        """

        if not flag:  # update using prediction
            self.b = self.lastResult
        else:  # update using detection
            self.b = b
        C = np.dot(self.A, np.dot(self.P, self.A.T)) + self.R
        K = np.dot(self.P, np.dot(self.A.T, np.linalg.inv(C)))

        self.u = np.round(self.u + np.dot(K, (self.b - np.dot(self.A,
                                                              self.u))))
        self.P = self.P - np.dot(K, np.dot(C, K.T))
        self.lastResult = self.u
        return self.u

class KalmanFilter_cv(object):
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2, 0)
        self.state = np.array([0, 0, 0, 0], dtype='float64')  # initial position
        self.kalman.transitionMatrix = np.array([[1., 0., .1, 0.],
                                            [0., 1., 0., .1],
                                            [0., 0., 1., 0.],
                                            [0., 0., 0., 1.]])
        self.kalman.measurementMatrix = 1. * np.eye(2, 4)
        self.kalman.processNoiseCov = 1e-5 * np.eye(4, 4)
        self.kalman.measurementNoiseCov = 1e-3 * np.eye(2, 2)
        self.kalman.errorCovPost = 1e-1 * np.eye(4, 4)
        self.kalman.statePost = self.state
    def predict(self):
        self.prediction = self.kalman.predict()
    def correct(self,p_scalar):
        #print("p shape ",np.shape(p_scalar))
        c,r,w,h = p_scalar
        if w != 0 and h != 0:
            self.state = np.array([c + w / 2, r + h / 2, 0, 0], dtype='float64')
            # kalman.statePost = state
            measurement = (np.dot(self.kalman.measurementNoiseCov, np.random.randn(2, 1))).reshape(-1)
            measurement = np.dot(self.kalman.measurementMatrix, self.state) + measurement
            posterior = self.kalman.correct(measurement)
            pos = (posterior[0], posterior[1])
        else:
            measurement = (np.dot(self.kalman.measurementNoiseCov, np.random.randn(2, 1))).reshape(-1)
            measurement = np.dot(self.kalman.measurementMatrix, self.state) + measurement
            pos = (self.prediction[0], self.prediction[1])
        process_noise = np.sqrt(self.kalman.processNoiseCov[0, 0]) * np.random.randn(4, 1)
        self.state = np.dot(self.kalman.transitionMatrix, self.state) + process_noise.reshape(-1)
        return pos
