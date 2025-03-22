import numpy as np
import os
import multiprocessing
import time
import traceback

from .face_store import FaceStore
from .face_embedding_modules.arcface import ARCFeatureExtractor 
from .face_embedding_modules.edgeface import  EFFeatureExtractor

class FaceVerifier:
    def __init__(self, threshold_stage1=0.5, threshold_stage2=0.6, debug=False):
        self.threshold_stage1 = threshold_stage1
        self.threshold_stage2 = threshold_stage2
        face_store_dir = os.environ["EMBEDDING_STORE_DIRECTORY"]
        print(face_store_dir)
        self.face_db_store1 = FaceStore(face_store_dir+"stage1_embeddings/")
        self.face_db_store2 = FaceStore(face_store_dir+"stage2_embeddings/")

        #instanciate a stage 1 and 2 feature extractor
        self.stage1_features = EFFeatureExtractor()
        self.stage2_features = ARCFeatureExtractor()
        #self.stage2_features = DLFeatureExtractor()

        self.debug = debug

    def process_face(self, frame, box):
        '''
        Takes in the frame/image and the bouding box of the face.
        Then calculates similarity scores of all bases in the database
        using the smaller model edgeface (also refered to stage 1). if the
        results are over a certain threshold the face is rechecked by the bigger model (stage 2)
        for more confidence.
        '''
        embedding = self.stage1_features.extract(frame, box)

        results = []
        recheck = False
        for idx, face in enumerate(self.face_db_store1.stored_faces):
            similarity_score = self.stage1_features.similarity(embedding, face)
            results.append([similarity_score, self.face_db_store1.mappings[idx][1], recheck])

        # Sort from most smallest similarity
        temporary_results = sorted(results, key=lambda x: x[0], reverse=True)
        
        '''
        1. List of faces similary scores per face is sorted
        2. Select the first (biggest as its sorted) score and its value
        3. If amount of face similarities that are over threshold are more than 0 then recheck with stage 2
        '''
        if len([result for result in results if result[0] > self.threshold_stage1]) > 0:
            recheck = True
            embedding = self.stage2_features.extract(frame, box)
            for idx, face in enumerate(self.face_db_store2.stored_faces):
                similarity_score = self.stage2_features.similarity(embedding, face)
                results[idx] = [similarity_score, self.face_db_store2.mappings[idx][1], recheck]
                
        results = sorted(results, key=lambda x: x[0], reverse=True)

        # If list is empty, early return
        if len(results) == 0:
            return

        # If similarity is blow threshold then face is unkown
        if (highest_similarity := results[0][0]) > self.threshold_stage2:
            return [highest_similarity, results[0][1], recheck]
        return -1

    def store_person(self, img, box, name):
        # Compute embedding
        embedding1 = self.stage1_features.extract(img, box)
        embedding2 = self.stage2_features.extract(img, box)
        try:
            self.face_db_store1[name] = embedding1
            self.face_db_store2[name] = embedding2
        except ValueError as e:
            raise ValueError(f"Failed to store Embedding: {e}")   

'''
The FaceVerifierProcess is a multiprocess wrapper for the face veritifer class.
Alongside being a wrapper it implements the queue that collects all faces to be verified
'''
class FaceVerifierProcess:
    def __init__(self, threshold_stage1=0.5, threshold_stage2=0.6, max_queued_per_id=3):
        self.threshold_stage1 = threshold_stage1
        self.threshold_stage2 = threshold_stage2
        self.max_queued_per_id = max_queued_per_id
        
        self.input_queue = multiprocessing.Queue()
        self.output_queue = multiprocessing.Queue()
        
        # Track incoming verifications by face ID
        self.queued_verifications = {}
        self.lock = multiprocessing.Lock()
        
        self.process = multiprocessing.Process(name="face_verifier", target=self.run, 
                                                args=(self.input_queue, self.output_queue), daemon=True)
        self.process.start()


    def start(self):
        if not self.process.is_alive():
            self.process.start()

    # Called when event analysis was ended 
    def reset(self):
        # Clear the queues
        try:
            while True:
                self.input_queue.get_nowait()
        except:
            pass
        try:
            while True:
                self.output_queue.get_nowait()
        except:
            pass
        
        with self.lock:
            self.queued_verifications.clear()


    def run(self, input_queue, output_queue):
        from .face_verifier import FaceVerifier

        verifier = FaceVerifier(
            threshold_stage1=self.threshold_stage1,
            threshold_stage2=self.threshold_stage2,
            debug=False
        )

        while True:
            # Process face and place results in the queue
            try:
                try:
                    task = input_queue.get()
                    if task is None:
                        break
                    frame, box, key, event_id = task
                    result = verifier.process_face(frame, box)
                    output_queue.put((key, result, event_id))
                except multiprocessing.queues.Empty:
                    pass
            except Exception as e:
                print(f"Error in processing queue item: {e}")

    # Called by event analyser to add a detection to the queue
    def submit_task(self, frame, box, key, event_id):
        '''
        If the same face id has already be queued n times then reject by returning false. This is important to
        keep old unidentifiable detection groups form blocking new identifiable detection groups
        '''
        with self.lock:
            if self.queued_verifications.get(key, 0) >= self.max_queued_per_id:
                return False
            self.queued_verifications[key] = self.queued_verifications.get(key, 0) + 1
        
        self.input_queue.put((frame, box, key, event_id, ))
        return True

    # Called by event analyser to get a detection from the queue
    def get_result(self):
        try:
            # Setting block false is important here
            result = self.output_queue.get(block=False)
            if result is not None:
                key = result[0]
                with self.lock:
                    self.queued_verifications[key] = max(0, self.queued_verifications.get(key, 0) -1)
            return result
        except multiprocessing.queues.Empty:
            return None

    def stop(self):
        self.input_queue.put(None)
        self.process.join()
