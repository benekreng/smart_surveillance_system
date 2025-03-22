from collections import defaultdict
import random
import time
import multiprocessing
import random
import queue
from typing import List, Union
import traceback

'''
Dummy Detector and Verifier for unit testing event analyser
'''
class DummyDetectorAndVerifier():
    def __init__(self):
        self.simulate_prossing_delay = False
        
        self.num_of_face_detections = 9
        self.num_of_people_in_scene = 3
        diff = abs(self.num_of_people_in_scene - self.num_of_face_detections)
        self.num_re_detections = diff
        # Num of re detections of which undefined
        self.num_unidentified = diff // 2
        
        self.num_of_frames = 10
        self.face_bboxes = 10
        self.dummy_detections = []
        self.idx = 0

        # Of these which are should be identified its index as its id 
        self.identities_as_bboxes = None
        
        # All bounding boxes
        self.bboxes = []
        self.generate_detections()

        # Setup face verifier process 
        self.in_queue = multiprocessing.Queue()
        self.out_queue = multiprocessing.Queue()
        self.process = multiprocessing.Process(target=self._verifier_worker, args=(self.in_queue, self.out_queue, self.bboxes), daemon=True)
        self.process.start()
        

    def generate_detections(self, dets_definition = [
                                        # every column refers to one identity.
                                        #  first digit > -1 is the trackin id added by the sort algo
                                        # -1 means no detetion happend
                                        # .0 means detection is unidentified
                                        # .1 means detection is identifid
                                        
                                        #    |> tracking ids referring to person 1 in the scene
                                        #    |     |> person 2
                                        #    |     |     |> person 3   |> Same for body detetections
                                        #    V     V     V             V
                                        [[0.0,  1.1, -1  ], [0.0,  -1,   -1  ]],
                                        [[0.0,  1.1, -1  ], [0.0,   1.0, -1  ]],
                                        [[0.0,  1.1,  2.0], [0.0,   1.0, -1  ]],
                                        [[0.1,  1.1,  2.0], [0.0,   1.0,  2.0]],
                                        [[0.1,  1.1,  2.1], [0.0,   1.0,  2.0]],
                                        [[0.1,  1.1,  2.0], [0.0,   1.0,  2.0]],
                                        [[0.1,  3.1, -1  ], [0.0,   1.0,  2.0]],
                                        [[0.1,  3.1, -1  ], [0.0,  -1,   -1  ]],
                                        [[4.0,  3.0, -1  ], [0.0,   1.0,  3.0]],
                                        [[4.1,  3.0, -1  ], [4.0,   1.0,  3.0]],
                                        [[5.1,  3.1, -1  ], [4.0,   1.0,  3.0]],
                                        [[5.1,  3.1, -1  ], [4.0,   1.0, -1  ]],
                                        [[5.0,  3.1, -1  ], [4.0,   1.0, -1  ]],
                                        [[5.0,  3.0, -1  ], [4.0,   1.0, -1  ]],
                                        [[6.1,  3.0, -1, 7.0], [4.0,   1.0, -1  ]],
                                        [[6.1,  3.0, -1, 7.0], [4.0,  -1,   -1  ]],
                                        [[6.1, -1,   -1, 7.0], [4.0,  -1,   -1  ]],
                                        [[ -1, -1,   -1  ], [-1,   -1,   -1  ]],
                                    ]):

        # Reset all detections
        self.reset()
        '''
        we create dummy bboxes. Unlike in the real senario, all the dummy bboxes that refer to the same person
        are exactly the same. This makes it easy for us to implement a dummy face verifier that identifies

        array holding the truth values to which person which box belongs
        [:self.num_of_people_in_scene] all refer to a different person
        [self.num_of_people_in_scene:] refer to bbox which could not be identified
        '''
    
        ### Creating unique boxes

        # Generating fake bboxes and their corresponding identifiy which our dummy face verifier can identify
        for idx in range(self.num_of_people_in_scene + 1):
            #Generate x1, y1, x2, y2 box
            x_step = 1/(self.num_of_face_detections +1)
            #0.25
            x1 = round(x_step * idx, 2)
            y1 = 0
            x2 = round(x1 + x_step, 2)
            y2 = 1
            bbox = [x1, y1, x2, y2]
            self.bboxes.append(bbox)

        self.identities_as_bboxes = self.bboxes[:self.num_of_people_in_scene]
        unidentified_bbox = self.bboxes[self.num_of_people_in_scene]

        for face_dets, body_dets in dets_definition:
            face_dets_per_frame = []
            for idx, ident in enumerate(face_dets):
                #if -1 nothing detected
                if ident == -1:
                    continue

                ID = int(ident)
                # If decimal place is .1 person was succefully identified
                if 0.1 == round(ident%1,2):
                    face_dets_per_frame.append([self.identities_as_bboxes[idx], ID])
                    continue

                # Decimal place is .0, face detected but could not be identified
                face_dets_per_frame.append([unidentified_bbox, ID])
                
            body_dets_per_frame = []
            for idx, ident in enumerate(body_dets):
                rand_bbox = [round(random.random(), 2) for _ in range(4)]
                if ident == -1:
                    continue

                ID = int(ident)
                body_dets_per_frame.append([rand_bbox, ID])

            self.dummy_detections.append([face_dets_per_frame, body_dets_per_frame])

        return self.dummy_detections

    def process_frame(self):
        if self.simulate_prossing_delay:
            time.sleep(0.12)
        
        if self.idx < len(self.dummy_detections):
            det = self.dummy_detections[self.idx]
            self.idx += 1
            return det[0], det[1]

        return [], []

    def face_verifier(self, bbox_to_id: List[float]) -> None:
        assert len(bbox_to_id) == 4, "bouding box must be 1D array of 4 floats (x1, y1, x2, y2)"
        for idx, bbox in enumerate(self.bboxes):
            if bbox_to_id == bbox:
                # Bbox at index=self.num_of_people_in_scene means person could not be defined
                if idx == self.num_of_people_in_scene:
                    return -1
                return idx
        return -1


    # Simulate face verifier process
    def _verifier_worker(self, in_queue, out_queue, bboxes):
        self.bboxes = bboxes
        while True:
            try:
                task = in_queue.get()
                if task is None:
                    break

                frame, box, key = task
                assert len(box) == 4, "bouding box must be 1D array of 4 floats (x1, y1, x2, y2)"

                # Simulate processing time
                if self.simulate_prossing_delay:
                    processing_time = random.uniform(0.5, 0.7)
                    time.sleep(processing_time)

                try:
                    #result = random.choice([0, 1])
                    result = self.face_verifier(box)
                    assert isinstance(result, int)
                except Exception as e:
                    traceback.print_exc()
                    continue
                    
                out_queue.put((key, result))
            except Exception as e:
                out_queue.put(("ERROR", str(e)))    

    def submit_task(self, frame, box, key):
        self.in_queue.put((frame, box, key))

    def get_result(self, block=False):
        try:
            return self.out_queue.get(block=block, timeout=0.1)
        except queue.Empty:
            return None
    
    def reset(self):
        self.idx = 0
        self.dummy_detections = []
        self.bboxes = []
            
    def stop(self):
        self.bboxes = []
        self.in_queue.put(None)
        self.process.join()