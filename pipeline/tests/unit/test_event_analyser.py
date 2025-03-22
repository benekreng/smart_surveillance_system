import pytest

from pipeline.dummy_detector_verifier import DummyDetectorAndVerifier
from pipeline.event_analyser import EventAnalyser 
import time


'''
This test tests the event analyser using the dummy_detector_verifier
'''
class TestEventAnalyser:
    @classmethod
    def setup_class(cls):
        cls.dummy_verifier = DummyDetectorAndVerifier()
        cls.event_analyser = EventAnalyser(cls.dummy_verifier)

    def teardown_class(cls):
        cls.dummy_verifier.stop()
        cls.event_analyser.reset_scene()

    def test_more_bodies_1(cls):
        custom_detections = [
            [[0.0,  1.1, -1  ], [0.0,  -1,   -1  ]],#Max bodies > max faces = Danger
            [[0.0,  1.1, -1  ], [0.0,   1.0, -1  ]],#|
            [[0.0,  1.1,  2.0], [0.0,   1.0, -1  ]],#V
            [[0.1,  1.1,  2.0], [0.0,   1.0,  2.0,   7.0]],
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
            [[6.1,  3.0, -1 ], [4.0,   1.0, -1  ]],
            [[6.1,  3.0, -1], [4.0,  -1,   -1  ]],
            [[6.1, -1,   -1, 7.0], [4.0,  -1,   -1  ]],
            [[ -1, -1,   -1  ], [-1,   -1,   -1  ]],
        ]
        cls.dummy_verifier.generate_detections(custom_detections)
        
        for i in range(len(custom_detections)):
            face_bbs_ids, body_bbs_ids = cls.dummy_verifier.process_frame()
            frame = None
            cls.event_analyser.update(frame, face_bbs_ids, body_bbs_ids)

        # Wait for the dummy face verifier to verify faces
        time.sleep(5)
        
        frame = None
        cls.event_analyser.update(frame, face_bbs_ids, body_bbs_ids)
        cls.event_analyser.evaluate_scene()
        cls.event_analyser.reset_scene()

        # Strict eval will evaluate this as NOT safe because max bodies != max identified faces
        assert not cls.event_analyser.strict_eval_safe, "Strict evaluation should not have evaluated Safe!"
        # Relaxed eval will evaluate this as safe because at least one known person was present
        assert cls.event_analyser.relaxed_eval_safe, "Relaxed evaluation should have evaluated Safe!"
        
    def test_more_bodies_2(cls):
        custom_detections = [
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
            [[6.1,  3.0, -1 ], [4.0,   1.0, -1  ]],
            [[6.1,  3.0, -1], [4.0,  -1,   -1  ]],
            [[6.1, -1,   -1, 7.0], [4.0,  -1,   -1  ]],
            [[ -1, -1,   -1  ], [-1,   -1,   -1  ]],
        ]
        cls.dummy_verifier.generate_detections(custom_detections)
        
        for i in range(len(custom_detections)):
            face_bbs_ids, body_bbs_ids = cls.dummy_verifier.process_frame()
            frame = None
            cls.event_analyser.update(frame, face_bbs_ids, body_bbs_ids)

        # Wait for the dummy face verifier to verify faces
        time.sleep(5)
        
        frame = None
        cls.event_analyser.update(frame, face_bbs_ids, body_bbs_ids)
        cls.event_analyser.evaluate_scene()
        cls.event_analyser.reset_scene()

        assert cls.event_analyser.strict_eval_safe, "Strict evaluation should have evaluated Safe!"
        assert cls.event_analyser.relaxed_eval_safe, "Relaxed evaluation should have evaluated Safe!"
         
    def test_more_bodies_3(cls):
        custom_detections = [
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
            [[6.1,  3.0, -1 ], [4.0,   1.0, -1  ]],
                        #face unidentified for more than 2 times, strict should evaluate danger 
                        #|
                        #V
            [[6.1,  3.0, 7.0], [4.0,  -1,   -1  ]],
            [[6.1, -1,   7.0], [4.0,  -1,   -1  ]],
            [[ -1, -1,   -1  ], [-1,   -1,   -1  ]],
        ]
        cls.dummy_verifier.generate_detections(custom_detections)
        
        for i in range(len(custom_detections)):
            face_bbs_ids, body_bbs_ids = cls.dummy_verifier.process_frame()
            frame = None
            cls.event_analyser.update(frame, face_bbs_ids, body_bbs_ids)

        #wait for the dummy face verifier to verify faces
        time.sleep(5)
        
        frame = None
        cls.event_analyser.update(frame, face_bbs_ids, body_bbs_ids)
        cls.event_analyser.max_undetected = 1
        cls.event_analyser.evaluate_scene()
        cls.event_analyser.reset_scene()

        assert not cls.event_analyser.strict_eval_safe, "Strict evaluation should not have evaluated Safe!"
        assert cls.event_analyser.relaxed_eval_safe, "Relaxed evaluation should have evaluated Safe!"
       
