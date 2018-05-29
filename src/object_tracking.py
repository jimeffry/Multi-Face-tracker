'''
    File name         : object_tracking.py
    File Description  : Multi Object Tracker Using Kalman Filter
                        and Hungarian Algorithm
    Author            : Srini Ananthakrishnan
    Date created      : 07/14/2017
    Date last modified: 07/16/2017
    Python Version    : 2.7
'''

# Import python libraries
import cv2
import copy
#from detectors import Detectors
from tracker import Tracker
import argparse
import numpy as np
from confg import config
from detector.get_faces import FaceDetector

def args():
    parser = argparse.ArgumentParser(description="Kalman and Hungarian Multi Obgect Detect")
    parser.add_argument('--file_in', type=str,default='None',\
                        help="video file input path")
    parser.add_argument('--key_num', type=int,default=1,\
                        help="every num frame to update detect")
    return parser.parse_args()

class Detector(object):
    def __init__(self,model_path):
        self.model_path = model_path
        self.detector_ = self.load_model()
    def load_model(self):
        detection_model = cv2.CascadeClassifier(self.model_path)
        print("load model over")
        return detection_model
    def detect_face(self,img,min_size=50,max_size=200):
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = self.detector_.detectMultiScale(gray_image, 1.3, 5,minSize=(min_size,min_size),maxSize=(max_size,max_size))
        #results = detector.detect_face(img)
        boxes = []
        for face_coordinates in faces:
            boxes.append(face_coordinates)
        if len(boxes)>0:
            return np.array(boxes)
        else:
            return None

def add_label(img,bbox,color=(255,0,0)):
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
        score_label = str(np.int(bbox[i,4]))
        #score_label = str(label)
        size = cv2.getTextSize(score_label, font, font_scale, thickness)[0]
        if y1-int(size[1]) <= 0:
            #cv2.rectangle(img, (x1, y2), (x1 + int(size[0]), y2+int(size[1])), color)
            cv2.putText(img, score_label, (x1,y2+size[1]), font, font_scale, color, thickness)
        else:
            #cv2.rectangle(img, (x1, y1-int(size[1])), (x1 + int(size[0]), y1), (255, 0, 0))
            cv2.putText(img, score_label, (x1,y1), font, font_scale, color, thickness)

def main(file_in,key_num):
    """Main function for multi object tracking
    Usage:
        $ python2.7 objectTracking.py
    Pre-requisite:
        - Python2.7
        - Numpy
        - SciPy
        - Opencv 3.0 for Python
    Args:
        None
    Return:
        None
    """

    # Create opencv video capture object
    if file_in == 'None':
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(file_in)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    f_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    model_path =  "../models/haarcascade_frontalface_default.xml"
    # Create Object Detector
    #detector = Detectors()
    #detector = Detector(model_path)
    prefix = ["../models/MTCNN_bright_model/PNet_landmark/PNet", \
            "../models/MTCNN_bright_model/RNet_landmark/RNet", \
            "../models/MTCNN_bright_model/ONet_landmark/ONet"]
    detector = FaceDetector(prefix)
    # Create Object Tracker
    tracker = Tracker(dist_thresh=150, max_frames_to_skip=5, max_trace_length=50, trackIdCount=0,w=480,h=640)
    # Variables initialization
    skip_frame_count = 0
    track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                    (0, 255, 255), (255, 0, 255), (255, 127, 255),
                    (127, 0, 255), (127, 0, 127)]
    pause = False
    cnt_frame = 0
    # Infinite loop to process video frames
    cv2.namedWindow("Tracking")
    cv2.moveWindow("Tracking",1400,10)
    cv2.namedWindow("Original")
    cv2.moveWindow("Original",100,10)
    centers = []
    key_frame = 0
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        #print("frame ",f_w,frame.shape)
        if f_w != 1280:
            frame = cv2.resize(frame,(1280,720))
        if not ret:
            print("video open failed")
            break
        # Make copy of original frame
        orig_frame = copy.copy(frame)
        # Skip initial frames that display logo
        '''
        if (skip_frame_count < 15):
            skip_frame_count += 1
            continue
        '''
        cnt_frame +=1
        # Detect and return centeroids of the objects in the frame
        if cnt_frame ==1:
            #bboxes = detector.detect_face(frame)
            bboxes = detector.get_face(frame)
            centers = []
            if bboxes is not None:
                key_frame = 1
                add_label(orig_frame,bboxes,(0,0,255))
                #centers = []
                bboxes_pre = bboxes
                print("detect box num ------------------------------",len(bboxes))
                c_array = np.vstack([bboxes[:,0]+bboxes[:,2]/2, bboxes[:,1]+bboxes[:,3]/2])
                c_array = c_array.T
                #print("face shape ",c_array.shape)
                #print(c_array)
                if config.tracker_cv:
                    bboxes[:,0] = bboxes[:,0] + bboxes[:,2]/2
                    bboxes[:,1] = bboxes[:,1] + bboxes[:,3]/2
                    centers = bboxes[:,:4]
                else:
                    for (x,y) in c_array:
                        centers.append(np.array([[x], [y]],dtype=np.int))
        if cnt_frame == key_num:
            cnt_frame = 0
        # If centroids are detected then track them
        if (len(centers) > 0):
            # Track object using Kalman Filter
            tracker.Update(centers,bboxes[:,2:4],key_frame)
            key_frame = 0
            # For identified object tracks draw tracking line
            # Use various colors to indicate different track_id
            box_rect = []
            if config.Debug:
                print("tracker num  ",len(tracker.tracks))
                print("assign num ",len(tracker.assign_out))
            assign_lenth = len(tracker.assign_out)
            #for i in range(len(tracker.tracks)):
            #for i in range(len(bboxes)):
            if len(tracker.del_tracks):
                for id in tracker.del_tracks:
                    if id < len(centers):
                        del centers[id]
                        print("----------del a bbox",len(centers))
            if len(centers) <1 :
                continue
            #for i in range(len(centers)):
            for i in range(len(bboxes)):
                #print("ind ",i)
                id_label = tracker.tracks[i].track_id
                if i < assign_lenth:
                    #print("as ",i)
                    face_id = tracker.assign_out[i]
                    w = bboxes_pre[face_id][2]
                    h = bboxes_pre[face_id][3]
                    #top_x = tracker.predict_center[i][0] - w/2
                    #top_y = tracker.predict_center[i][1] - h/2
                    top_x = tracker.tracks[i].prediction[0] - w/2
                    top_y = tracker.tracks[i].prediction[1] - h/2
                    box_split = np.hstack([top_x,top_y,w,h,id_label])
                    box_rect.append(box_split)
                else:
                    box_split = np.hstack([bboxes_pre[i,:4],id_label])
                    box_rect.append(box_split)
                if (len(tracker.tracks[i].trace) > 1):
                    for j in range(len(tracker.tracks[i].trace)-1):
                        # Draw trace line
                        if config.tracker_cv:
                            x1 = tracker.tracks[i].trace[j][0]
                            y1 = tracker.tracks[i].trace[j][1]
                            x2 = tracker.tracks[i].trace[j+1][0]
                            y2 = tracker.tracks[i].trace[j+1][1]
                        else:
                            x1 = tracker.tracks[i].trace[j][0][0]
                            y1 = tracker.tracks[i].trace[j][1][0]
                            x2 = tracker.tracks[i].trace[j+1][0][0]
                            y2 = tracker.tracks[i].trace[j+1][1][0]
                        clr = tracker.tracks[i].track_id % 9
                        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                 track_colors[clr], 2)
            #centers = tracker.predict_center
            # Display the resulting tracking frame
            #box_rect = np.array([box_rect],dtype=np.int)
            #box_rect = np.vstack(box_rect)
            box_rect = np.asarray(box_rect)
            if config.Debug:
                print("bbox_label shape ",box_rect.shape)
            add_label(frame,box_rect)
            cv2.imshow('Tracking', frame)
        # Display the original frame
        cv2.imshow('Original', orig_frame)
        # Slower the FPS
        cv2.waitKey(10)
        # Check for key strokes
        k = cv2.waitKey(50) & 0xff
        if k == 27:  # 'esc' key has been pressed, exit program.
            break
        if k == 112:  # 'p' has been pressed. this will pause/resume the code.
            pause = not pause
            if (pause is True):
                print("Code is paused. Press 'p' to resume..")
                while (pause is True):
                    # stay in this loop until
                    key = cv2.waitKey(30) & 0xff
                    if key == 112:
                        pause = False
                        cnt_frame = 0
                        print("Resume code..!!")
                        break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # execute main
    parm = args()
    file_in = parm.file_in
    key_num = parm.key_num
    main(file_in,key_num)
