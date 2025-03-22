import unittest
import time
from pipeline.dummy_detector_verifier import DummyDetectorAndVerifier

class TestDummyFaceVerifierProcess(unittest.TestCase):
    def setUp(self):
        # Initialize before each test.
        self.verifier = DummyDetectorAndVerifier()
        self.verifier.generate_detections()

    def tearDown(self):
        # Initialize before each test.
        # Close process after test is done
        self.verifier.stop()

    def test_submit_and_receive_result(self):
        #Test to submit a single task
        frame = None
        box = [50, 50, 100, 100]
        key = 1

        self.verifier.submit_task(frame, box, key)

        result = None
        # Retry 5 times after waiting for delay
        for _ in range(5):
            result = self.verifier.get_result(block=True)
            if result:
                break
            time.sleep(0.3)
        
        self.assertIsNotNone(result, "No result received from the verifier")
        key_received, verification_result = result
        self.assertEqual(key_received, key, "Task id does not match")
        self.assertIsInstance(verification_result, int, f"Result {verification_result} is not an integer")
        self.assertGreater(verification_result,-2, f"Result {verification_result} is not greater than -1")

    #Test submitting multiple tasks
    def test_multiple_tasks(self):
        #place holder frame
        frame = None
        bounding_boxes = [
            [0.1, 0.3, 0.4, 0.1],
            [0.2, 0.3, 0.1, 0.8],
            [0.7, 0.8, 0.9, 0.8],
        ]

        tasks = list(range(len(bounding_boxes)))

        # Submit tasks
        for i, box in enumerate(bounding_boxes):
            self.verifier.submit_task(frame, box, key=tasks[i])

        # Collect results
        results = {}
        # Allow multiple attempts to recieve results
        for _ in range(10):
            response = self.verifier.get_result(block=True)
            if response:
                key_received, verification_result = response
                results[key_received] = verification_result
            # Stop after all tasks have been received
            if len(results) == len(bounding_boxes):
                break
            time.sleep(0.3)

        # Ensure we received all results
        self.assertEqual(len(results), len(bounding_boxes), "Not all tasks returned results")

        # Check all results are valid (0 or 1)
        for key in tasks:
            self.assertIsInstance(results[key], int, f"Result {results[key]} is not an integer")
            self.assertGreater(results[key], -2, f"Result {results[key]} is not greater than -1")

    def test_known_person_verification(self):
        # Test submitting a bounding box that belongs to a known person
        frame = None
        print("Generated bounding boxes:", self.verifier.bboxes)

        if not self.verifier.bboxes:
            self.fail("Bounding boxes were not generated properly")

        known_bbox0 = self.verifier.bboxes[0]
        key = 0
        self.verifier.submit_task(frame, known_bbox0, key)
        
        known_bbox1 = self.verifier.bboxes[1]
        key = 1
        self.verifier.submit_task(frame, known_bbox1, key)


        results = []
        found = 0
        while found < 2:
            result = self.verifier.get_result(block=True)
            
            if result:
                found += 1
                results.append(result)
            
            time.sleep(0.3)

        for result in results:
            print("THIS IS THE RESULT", result)
            self.assertIsNotNone(result, "No result received from the verifier")
            key_received, verification_result = result
            self.assertEqual(key_received, verification_result, "Task id does not match")