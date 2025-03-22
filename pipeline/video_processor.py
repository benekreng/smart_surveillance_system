from pipeline.face_detector import FaceDetector
from pipeline.body_detector import BodyDetection
from pipeline.SFSORT import SFSORT

import numpy as np
import time
import cv2

import warnings


class VideoProcessor:
    def __init__(self, face_verifier_process, face_detector):
        self.face_verifier_process = face_verifier_process 
        self.face_detector = face_detector
        self.obj_detector = BodyDetection(
            threads=3,
            #model_input_shape=[256, 256],
            #model_input_shape=[320, 320],
            model_input_shape=[160, 160],
            confidence_thres=0.60,
            iou_thres=0.45,
            debug=False
        )
        # Make face tracking tracking more lenient by adjusting arguments for SFSORT
        args = {
            "match_th_first": 0.67,
            "match_th_second": 0.97,
            "marginal_timeout": 12.0,
            "central_timeout": 19.0
        }
        self.face_mot_tracker = SFSORT(args)

        # Make body tracking tracking more lenient by adjusting arguments for SFSORT
        args = {
            "match_th_first": 0.66,
            "match_th_second": 0.95,
            "marginal_timeout": 9.0,
            "central_timeout": 15.0
        }
        # Initialize background subtractor and sfsort
        self.body_mot_tracker = SFSORT(args)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
        self.learning_rate = 0.05
        self.min_motion_area = 50
        self.every_nth_frame = 0
        
    def start_scene(self):
        self.event = True 
        self.start_event_time = time.time() 
    
    def keep_bg_sub_alive(self, frame):
        frame = cv2.resize(frame, (180, 120))
        _ = self.bg_subtractor.apply(frame, learningRate=self.learning_rate)

    # Periodically called by pipeline controller when no event is active, returns true if motion detected is over critical threshold
    def bg_subtraction(self, frame):
        # If critical mass of detections are reached then start further analysis
        # Artificially limit frame rate to 18fps
        time.sleep(0.05)
        frame = cv2.resize(frame, (180, 120))
        fg_mask = self.bg_subtractor.apply(frame, learningRate=self.learning_rate)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(fg_mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 

        # Check if the contour area exceeds motion threshold
        motion_detected = any(cv2.contourArea(c) > self.min_motion_area for c in contours)
        return motion_detected

    # Periodically called by pipeline controller when event is active (confirmed or not confirmed)
    def process_frame(self, frame):
        body_track_bbs_ids = []
        if (self.every_nth_frame%3) == 0: 
            body_boxes, body_scores = self.obj_detector.detect(frame)
            body_track_bbs_ids = self.body_mot_tracker.update(body_boxes, body_scores)

        self.every_nth_frame += 1
        '''
        WARNING: 
        Sfsort tracker returns inlcude a numpy array: larray [array([0.5083027 , 0.20966568, 0.55249416, 0.31232382]) 0].
        I modified the SFSORT source file to return a pure python array [[0,1,2,3],0]
        '''
        
        faces_boxes, faces_scores = self.face_detector.process_frame(frame)
        track_bbs_ids = self.face_mot_tracker.update(faces_boxes, faces_scores)
        return track_bbs_ids, body_track_bbs_ids