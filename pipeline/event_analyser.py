from collections import defaultdict
import pprint
import random
import time

'''
The event alyser State Manager is using state pattern. This is critical as the state transitions has to happen immediately 
within the same cycle (this is not the case when using contitinal statments). The pipeline controller required these immediate state
transitions fromt he event analyser to function correctly.
'''

# Event state base class
class EventState:
    def __init__(self, analyser):
        self.analyser = analyser

    def handle(self, face_dets=[], body_det=[], frame=None):
        raise NotImplementedError("Function has to be overriden")

# Do nothing
class IdleState(EventState):
    def handle(self, face_dets=[], body_det=[], frame=None):
        return self

# During event, periodically update analyser
class RunningState(EventState):
    def handle(self, face_dets=[], body_det=[], frame=None):
        self.analyser.analysis_time = time.time()
        self.analyser._update(face_dets, body_det, frame)
        return self

# Counts down analysis timer, periodically update analyser
class PostEventState(EventState):
    def handle(self, face_dets=[], body_det=[], frame=None):
        self.analyser._update(face_dets, body_det, frame)
        elapsed_time = time.time() - self.analyser.analysis_time
        if elapsed_time > self.analyser.max_analysis_time_post_end:
            return self.analyser.end_analysis_state.handle([], [], None)
        return self

# Logic that ecexutes after event end
class EndAnalysisState(EventState):
    def handle(self, face_dets=[], body_det=[], frame=None):
        print("End analysis state called!")
        # Force final update to collect verifications one last time
        self.analyser._update([], [])
        # Evaluate scene and reset
        self.analyser.evaluate_scene()
        self.analyser.print_evaluation_results()
        self.analyser.reset_scene()
        self.analyser.event_id += 1
        return self.analyser.idle_state

class EventAnalyser():
    def __init__(self, face_verifier_process, max_analysis_time, event_results_callback):
        self.event_results_callback = event_results_callback
        self.face_verifier_process = face_verifier_process
        self.event_id = 0
        
        self.idle_state = IdleState(self)
        self.running_state = RunningState(self)
        self.post_state = PostEventState(self)
        self.end_analysis_state = EndAnalysisState(self)
        self.start_scene()

        self.max_analysis_time_post_end = max_analysis_time
        self.strict_eval_safe = None
        self.relaxed_eval_safe = None
        self.post_event = False
        # Initially, we are in the idle state.
        self.state = self.idle_state
        
    def start_scene(self):
        self.faces = defaultdict(list)
        self.bodies = defaultdict(list)
        self.face_results = {}
        self.face_amt_eval = defaultdict(int)
        self.body_id_status = {}
        self.simultaneous_face_ids = []
        self.simultaneous_body_ids = []
        self.max_undetected = 1
        self.analysis_time = 0

    def reset_scene(self):
        self.face_verifier_process.reset()
        self.faces.clear()
        self.bodies.clear()
        self.face_results.clear()
        self.face_amt_eval.clear()
        self.body_id_status.clear()
        self.simultaneous_face_ids.clear()
        self.simultaneous_body_ids.clear()
        self.max_undetected = 1
        self.analysis_time = 0

    def update(self, face_dets=[], body_det=[], frame=None):
        self.state = self.state.handle(face_dets, body_det, frame)

    def new_event(self, face_dets=[], body_det=[], frame=None):
        # If new event is triggered while we're in post event mode, force end immediately
        if isinstance(self.state, PostEventState):
            self.state = self.end_analysis_state.handle([], [], None)

        # Otherwise just switch to running state
        self.state = self.running_state
        self.update(face_dets, body_det, frame)

    def event_ended(self, face_dets=[], body_det=[], frame=None):
        print("event ended called")
        if isinstance(self.state, RunningState):
            self.state = self.post_state
        self.update(face_dets, body_det, frame)

    # Console print of detection
    def print_evaluation_results(self):
        separator = "=" * 60
        print(separator)
        if self.strict_eval_safe:
            print("Strict Eval: No Unknown Person Present -> Safe")
        else:
            print("Strict Eval: Unknown Person Present -> Danger")
            
        if self.relaxed_eval_safe:
            print("Relaxed Eval: Known Person Present -> Safe")
        else:
            print("Relaxed Eval: Only Unknown Person Present -> Danger")

        print("\nFace Detection Results:")
        print(f"Face Dict: {self.face_results}")
        print("\nFace IDs:")
        print(f"{self.simultaneous_face_ids}")
        print("\nRescan Amounts:")
        print(f"{dict(self.face_amt_eval)}")
        print("\nBodies:")
        print(f"{self.simultaneous_body_ids}")
        print(separator)

    def _update(self, face_dets=[], body_det=[], frame=None):
        # Store face detections with id
        face_ids = []
        if face_dets:
            for det in face_dets:
                assert len(det[0]) == 4, "Array must be of format: [[x1, y1, x2, y2], id: int]"
                bbox, ID = det[0], det[1]
                face_ids.append(ID)
                self.faces[ID].append(bbox)
                self.face_results.setdefault(ID, -1)
            self.simultaneous_face_ids.append(face_ids)

        # Only body detection ids. Just the quantity per frame is needed
        body_ids = []
        if body_det:
            for det in body_det:
                bbox, ID = det[0], det[1]
                body_ids.append(ID)
            self.simultaneous_body_ids.append(body_ids)

        # Get results from the face verifier process.
        while True:
            response = self.face_verifier_process.get_result()
            if response is None:
                break
            key, result, result_event_id = response

            # If result is not -1 then store result
            if result_event_id == self.event_id:
                # Increment face eval count
                self.face_amt_eval[key] += 1
                if result == "unknown":
                    result = -1
                if result is None or result == None:
                    result = -1

                # making sure to ignore analysis results form previous events
                if result != -1:
                    self.face_results[key] = result

        # Queue any bboxes that have not yet been identified
        queue = []
        for key in self.faces:
            # For each face id, add one new detection if face id status still undetected (-1)
            if self.faces[key] and self.face_results[key] == -1:
                # Append to local buffer
                queue.append([self.faces[key][0], key])
                # Remove pushed bbox from detections
                self.faces[key] = self.faces[key][1:]

        # Submit queued tasks (detections) to face verifier
        for face_det in queue:
            bbox, key = face_det[0], face_det[1]
            assert len(bbox) == 4, "Array must be of format: [x1, y1, x2, y2]"
            assert isinstance(key, int), "ID must be an int"

            '''
            The face verifier will reject a face if its id is already present more than n times (max_pending_per_face).
            This ensures older (perhaps undetectable faces) dont accumulate and block analysis of recent scanned faces.
            '''
            
            # Submit the task and check if it was accepted
            task_accepted = self.face_verifier_process.submit_task(frame, bbox, key, self.event_id)
            
            # If rejected place the detection back into detections object
            if not task_accepted:
                self.faces[key].insert(0, bbox)


    def evaluate_scene(self):
        '''
        Compute relaxed evaluation. Deems event save if at least one face id group could be identified.
        '''
        self.relaxed_eval_safe = False

        # If at least known person was detected situation is safe
        for key in self.face_results:
            if self.face_results[key] != -1:
                self.relaxed_eval_safe = True
        
        # Evaluate safety based on face and body counts.
        max_faces = max((len(frame) for frame in self.simultaneous_face_ids), default=0)
        max_bodies = max((len(frame) for frame in self.simultaneous_body_ids), default=0)

        '''
        Compute strict evaluation
        '''
        self.strict_eval_safe = True

        # Dictionary that holds the number of detections for each face
        face_dets_amounts = defaultdict(int)
        for frame in self.simultaneous_face_ids:
            for face in frame:
                face_dets_amounts[face] += 1

        # If more bodies than faces were detected situation is unsafe
        if max_bodies > max_faces:
            self.strict_eval_safe = False

        # if one tracking id could not be detected situation is unsafe
        for key in self.face_results:
            if self.face_results[key] == -1:
                '''
                Low FPS Mode:
                If unknown face was detected more than max_undetected times and the total facesthe total then raise alarm.
                If amount of unidentified (-1) faces is less than max_undetected and face detections form all ID's is more than 6
                then ignore detection as its likely a face detection seeprated from another group of detections.
                This setting effecively decreases false positives by a good amount when the system runs at a low frame rate.

                WARNING: 
                This setting is controversial as it can produce false negatives.

                Note:
                In 300 test cases (on a raspberry pi 5 running at 8-18fps), no false negative where obseverd with this setting.
                '''
                low_fps_mode = False
                if low_fps_mode:
                    total_face_detections = sum(face_dets_amounts.values())

                    if face_dets_amounts[key] > self.max_undetected:
                        self.strict_eval_safe = False

                    elif total_face_detections < 6:
                        self.strict_eval_safe = False
                else:
                # Low fps mode off: if any face id group is unidentified strict evaluation deems event not safe
                    self.strict_eval_safe = False



        # Return evaluation and event details
        event_data = {
            "face_results": self.face_results,
            "bodies": self.simultaneous_body_ids,
            "simultaneous_face_ids": self.simultaneous_face_ids,
            "strict_eval_safe": self.strict_eval_safe,
            "relaxed_eval_safe": self.relaxed_eval_safe,
            "identification_scan_count": self.face_amt_eval,
        }
        
        # Call pipeline controller callback
        self.event_results_callback(self.event_id, event_data)