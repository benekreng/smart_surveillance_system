
from pipeline.debug import draw_detections, FPSCounter
from pipeline.event_analyser import EventAnalyser
from pipeline.video_processor import VideoProcessor
import db

import numpy as np
import time
import threading
import numpy as np
import asyncio
from IPython.display import display, clear_output
from PIL import Image
import cv2
import random
import traceback
import json
import csv

class PipelineController:
    # State constants
    STATE_WAIT_FOR_EVENT = 'state_wait_for_event'
    STATE_CONFIRM_EVENT = 'state_confirm_event'
    STATE_START_EVENT = 'state_start_event'
    STATE_EVENT_RUNNING = 'state_event_running'
    STATE_EVENT_END = 'state_event_end'

    def __init__(
        self, stream, 
        frame_buffer, 
        face_verifier_process, 
        face_detector, 
        db_async_session, 
        event_timeout, 
        max_analysis_time, 
        telegram_bot
    ):
        self.running = False
        self.face_verifier_process = face_verifier_process
        self.video_processor = VideoProcessor(self.face_verifier_process, face_detector)
        self.event_analyser = EventAnalyser(self.face_verifier_process, max_analysis_time, self.event_results_callback)
        self.frame_buffer = frame_buffer
        self.stream = stream
        #self.detection_store = detection_store
        self.fps_counter = FPSCounter(averaging_interval=1.0)

        self.db_async_session = db_async_session
        self.bot = telegram_bot

        self.state = self.STATE_WAIT_FOR_EVENT
        self.analysis_start_time = None

        self.frame_count = 0
        self.start_event_time = 0
        self.event_timeout = event_timeout
        self.frame = None
        self.body_bbs_ids = []
        self.face_bbs_ids = []
        self.event = True

    # Callback called when analysis finished by event analyser
    def event_results_callback(self, event_id, event_data):
        """Callback function for when event analysis is complete"""
        # Extract data from the event
        face_results = event_data.get('face_results', {})
        
        # Dictionary to store identity confidences
        identity_confidences = {}
        
        # Extract identities and their confidences
        for key, value in face_results.items():
            if isinstance(value, list):
                try:
                    confidence, identity, _ = value
                    if identity in identity_confidences:
                        identity_confidences[identity].append(float(confidence))
                    else:
                        identity_confidences[identity] = [float(confidence)]
                except Exception as e:
                    print(f"Error processing face result: {e}")
        
        # Get max confidence for each identity
        identity_results = []
        for identity, confidences in identity_confidences.items():
            max_conf = max(confidences)
            identity_results.append({
                "id": identity,
                "confidence": round(max_conf, 2)
            })

        # Get evaluation results
        strict_eval_safe = event_data.get('strict_eval_safe', False)
        relaxed_eval_safe = event_data.get('relaxed_eval_safe', False)
        
        # Helper function for converting numpy and other non-serializable types
        def make_serializable(obj):
            # numpy to native python
            if isinstance(obj, (np.ndarray, np.number)):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            else:
                return obj
        
        print(f"Event finished")
        print(f"Identities: {identity_results}")
        print(f"Strict evaluation: {'safe' if strict_eval_safe else 'alert'}")
        print(f"Relaxed evaluation: {'safe' if relaxed_eval_safe else 'alert'}")
        
        # Serialized data
        identities = json.dumps(identity_results)
        strict_eval_safe = strict_eval_safe
        relaxed_eval_safe = relaxed_eval_safe
        video_path = self.current_video_path
        face_results = json.dumps(make_serializable(face_results))
        simultaneous_face_ids = json.dumps(make_serializable(event_data.get('simultaneous_face_ids', [])))
        face_amt_eval = json.dumps(make_serializable(event_data.get('identification_scan_count', {})))
        simultaneous_body_ids = json.dumps(make_serializable(event_data.get('bodies', [])))

        # Make detection database entry
        try:
            asyncio.run(
                db.add_detection(
                    self.db_async_session,
                    identities=identities,
                    strict_eval_safe=strict_eval_safe,
                    relaxed_eval_safe=relaxed_eval_safe,
                    video_path=video_path,
                    face_results=face_results,
                    simultaneous_face_ids=simultaneous_face_ids,
                    face_amt_eval=face_amt_eval,
                    simultaneous_body_ids=simultaneous_body_ids
                )
            )
        except Exception as e:
            print(f"Error saving to db {e}")
       
        print("Event saved to database")
        try:
            print(f"Strict eval: {strict_eval_safe}, Relaxed eval: {relaxed_eval_safe}")
            message = self.bot.send_alert(identity_results, strict_eval_safe, relaxed_eval_safe)
        except Exception as e:
            print(f"Notification could not be send: {e}")

        # Writing results to csv for debug purposes
        # with open('results.csv', 'a', newline='') as csvfile:
        #     spamwriter = csv.writer(csvfile, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        #     spamwriter.writerow([event_id, face_results, strict_eval_safe, relaxed_eval_safe, simultaneous_face_ids, simultaneous_body_ids, face_amt_eval])

    
    '''
    STATES. Some states are called repeatedly whereas others 
    only run once between other states, serving as a transition.
    '''
    # Idle state
    def wait_for_event(self):
        self.read_frame()
        self.update_debug_stream()

        # When sufficient motion is detected move on event confirmation
        if self.motion_detected():
            self.confirm_event_time = time.time()
            self.stream.start_recording()
            self.state = self.STATE_CONFIRM_EVENT
            print("looking for event confirmation")

    # State which confirms an event by waiting for a body or face detection
    def confirm_event(self):
        self.read_frame()
        self.run_detections()
        self.update_debug_stream()

        # If body of face detected start event
        if len(self.face_bbs_ids) > 0 or len(self.body_bbs_ids) > 0:
            self.state = self.STATE_START_EVENT

        # If time runs out before event could be confirmed return back to idle state
        if time.time() - self.confirm_event_time > self.event_timeout:
            print("waiting for event")
            self.stream.discard_recording()
            #self.stream.stop_recording()
            self.state = self.STATE_WAIT_FOR_EVENT

    # Start event (transitional state)
    def start_event(self):
        self.start_event_time = time.time()
        # end event for analyser!
        self.event_analyser.new_event()
        #self.event_analyser.end_analysis()
        self.state = self.STATE_EVENT_RUNNING
        print("event started")

    '''
    Runs when event is live. 
    If no detection where made for a certain amount of time then end event
    '''
    def event_running(self):
        self.read_frame()
        self.run_detections()
        self.update_debug_stream()
        self.keep_bgs_alive()
        
        # If body is detected reset timeout timer
        if len(self.face_bbs_ids) > 0 or len(self.body_bbs_ids) > 0:
            self.start_event_time = time.time()

        # Stop event after timer runs out and no detections where detected
        if time.time() - self.start_event_time > self.event_timeout:
            self.state = self.STATE_EVENT_END
            print("event ended")

    # Event end state transition (transitional state)
    def event_end(self):
        self.event_analyser.event_ended()
        
        # Get the video file path and store it before stopping recording
        self.current_video_path = f"saved_clips/event_{self.event_analyser.event_id}_{int(time.time())}.mp4"
        self.stream.set_output_filename(self.current_video_path)
        self.stream.stop_recording()
        
        self.state = self.STATE_WAIT_FOR_EVENT
        print(f"Event ended, saved to {self.current_video_path}")

    # Thread running states continuosly
    def thread(self):
        try:
            while self.running:
                self.event_analyser.update(self.face_bbs_ids, self.body_bbs_ids, self.frame)
                # refresh event_analyser
                if self.state == self.STATE_WAIT_FOR_EVENT:
                    self.wait_for_event()
                elif self.state == self.STATE_CONFIRM_EVENT:
                    self.confirm_event()
                elif self.state == self.STATE_START_EVENT:
                    self.start_event()
                elif self.state == self.STATE_EVENT_RUNNING:
                    self.event_running()
                elif self.state == self.STATE_EVENT_END:
                    self.event_end()
    
        except KeyboardInterrupt:
            print("Pipeline thread stopped.")
    
    def read_frame(self):
        self.frame_count += 1
        ret, self.frame = self.stream.read()

    def run_detections(self):
        self.face_bbs_ids, self.body_bbs_ids = self.video_processor.process_frame(self.frame)

    def update_debug_stream(self):
        self.fps_counter.update()
        # Create live video feed with annotations
        try:
            if self.state != self.STATE_CONFIRM_EVENT and self.state != self.STATE_WAIT_FOR_EVENT:
                event_confirmed = True
            else:
                event_confirmed = False

            debug_frame = draw_detections(self.frame, self.body_bbs_ids, self.face_bbs_ids, self.fps_counter.get_fps(), 
                                            event_confirmed, self.event_analyser.event_id)
        except Exception as e:
            error_message = f"Error when drawing detections: {e}"
            debug_frame = frame
        self.frame_buffer.set(debug_frame)
    
    # Called when event is live to refresh MOG2 so it keeps learning
    def keep_bgs_alive(self):
        # every nth frame keep backbround subtractor alive to keep bg model from decaying
        #if (self.frame_count % 3) == 0:
        self.video_processor.keep_bg_sub_alive(self.frame)

    def motion_detected(self):
        motion_detected = self.video_processor.bg_subtraction(self.frame)
        return motion_detected