import os
import sys
import cv2
import numpy as np
import time
from kalman_filter import KalmanFilter_cv

def load_detection_model(model_path):
    detection_model = cv2.CascadeClassifier(model_path)
    print("load model over")
    return detection_model

def detect_face(img,detection_model):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detection_model.detectMultiScale(gray_image, 1.3, 5,minSize=(50,50),maxSize=(200,200))
    #results = detector.detect_face(img)
    boxes = []
    for face_coordinates in faces:
        boxes.append(face_coordinates)
    if len(boxes)>0:
        return np.array(boxes[0])
    else:
        return np.array([0,0,0,0])

def draw_box(img,box,color=(255,0,0)):
    (row,col,cl) = np.shape(img)
    #b = board_img(box,col,row)
    cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]+box[0]), int(box[3]+box[1])), color)

def add_label(img,bbox,label,color=(255,0,0)):
    num = bbox.shape[0]
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    font_scale =1
    thickness = 1
    for i in range(num):
        x1,y1,w,h = int(bbox[i,0]),int(bbox[i,1]),int(bbox[i,2]),int(bbox[i,3])
        x2 = x1 + w
        y2 = y1 + h
        cv2.rectangle(img,(x1,y1),(x2,y2),color,1)
        #score_label = str('{:.2f}'.format(bbox[i,4]))
        score_label = label
        size = cv2.getTextSize(score_label, font, font_scale, thickness)[0]
        if y1-int(size[1]) <= 0:
            #cv2.rectangle(img, (x1, y2), (x1 + int(size[0]), y2+int(size[1])), color)
            cv2.putText(img, score_label, (x1,y2+size[1]), font, font_scale, color, thickness)
        else:
            #cv2.rectangle(img, (x1, y1-int(size[1])), (x1 + int(size[0]), y1), (255, 0, 0))
            cv2.putText(img, score_label, (x1,y1), font, font_scale, color, thickness)

def kalman_filter_tracker(v,model_path,update_num):
    # Open output file
    frameCounter = 0
    # read first frame
    ret, frame = v.read()
    if not ret:
        print("open file failed")
        return -1
    # detect face in first frame
    detector = load_detection_model(model_path)
    c, r, w, h = detect_face(frame,detector)
    # Write track point for first frame
    pt = (0,   c + w/2.0, r + h/2.0)
    #output.write("%d,%d,%d\n" % pt)  # Write as 0,pt_x,pt_y
    frameCounter += 1
    kalman = KalmanFilter_cv()
    '''
    state = np.array([c + w / 2, r + h / 2, 0, 0], dtype='float64')  # initial position
    kalman.transitionMatrix = np.array([[1., 0., .1, 0.],
                                        [0., 1., 0., .1],
                                        [0., 0., 1., 0.],
                                        [0., 0., 0., 1.]])
    kalman.measurementMatrix = 1. * np.eye(2, 4)
    kalman.processNoiseCov = 1e-5 * np.eye(4, 4)
    kalman.measurementNoiseCov = 1e-3 * np.eye(2, 2)
    kalman.errorCovPost = 1e-1 * np.eye(4, 4)
    kalman.statePost = state
    '''
    while (1):
        # use prediction or posterior as your tracking result
        ret, frame = v.read()  # read another frame
        if not ret:
            print("read over")
            break

        img_width = frame.shape[0]
        img_height = frame.shape[1]
        '''
        def calc_point(angle):
            return (np.around(img_width / 2 + img_width / 3 * np.cos(angle), 0).astype(int),
                    np.around(img_height / 2 - img_width / 3 * np.sin(angle), 1).astype(int))
        '''
        # e.g. cv2.meanS    hift, cv2.CamShift, or kalman.predict(), kalman.correct()

        # use the tracking result to get the tracking point (pt):
        # if you track a rect (e.g. face detector) take the mid point,
        # if you track particles - take the weighted average
        # the Kalman filter already has the tracking point in the state vector
        frameCounter +=1
        t1 =time.time()
        if frameCounter ==1:
            det_box = detect_face(frame,detector)
            if det_box[2]!=0 and det_box[3]!=0:
                c, r, w, h = det_box
            bbox = np.array([[c, r, w, h]])
            add_label(frame,bbox,"real")
        t2 = time.time()-t1
        print("detect time: ",t2)
        t1 = time.time()
        kalman.predict()
        pos = 0
        #c, r, w, h = detect_one_face(frame)
        #draw_box(frame,(c,r,w,h))
        '''
        if w != 0 and h != 0:
            state = np.array([c + w / 2, r + h / 2, 0, 0], dtype='float64')
            # kalman.statePost = state
            measurement = (np.dot(kalman.measurementNoiseCov, np.random.randn(2, 1))).reshape(-1)
            measurement = np.dot(kalman.measurementMatrix, state) + measurement
            posterior = kalman.correct(measurement)
            pos = (posterior[0], posterior[1])
        else:
            measurement = (np.dot(kalman.measurementNoiseCov, np.random.randn(2, 1))).reshape(-1)
            measurement = np.dot(kalman.measurementMatrix, state) + measurement
            pos = (prediction[0], prediction[1])

        # display_kalman3(frame, pos, (c, r, w, h))
        process_noise = np.sqrt(kalman.processNoiseCov[0, 0]) * np.random.randn(4, 1)
        state = np.dot(kalman.transitionMatrix, state) + process_noise.reshape(-1)
        '''
        pos = kalman.correct([c,r,w,h])
        t3 = time.time()-t1
        t1 = time.time()
        print("tracking time: ",t3)
        #draw_box(frame,(pos[0],pos[1],w,h),color=(0,255,0))
        bbox = np.array([[pos[0]-w/2,pos[1]-h/2,w,h]])
        add_label(frame,bbox,"predict",color=(0,255,0))
        c,r,w,h = pos[0]-w/2,pos[1]-h/2,w,h
        pt = (frameCounter, pos[0], pos[1])
        '''
        if frameCounter != 256:
            output.write("%d,%d,%d\n" % pt) # Write as frame_index,pt_x,pt_y
        else:
            output.write("%d,%d,%d" % pt)  # Write as frame_index,pt_x,pt_y
        '''
        if frameCounter == update_num:
            frameCounter=0
        cv2.imshow("result",frame)
        cv2.waitKey(10)
        t4 = time.time()-t1
        print("total time: ",t4+t3+t2)
    #output.close()

if __name__ == '__main__':
    # read video file
    model_path = "../models/haarcascade_frontalface_default.xml"
    video = cv2.VideoCapture(0)
    kalman_filter_tracker(video,model_path, 5)
