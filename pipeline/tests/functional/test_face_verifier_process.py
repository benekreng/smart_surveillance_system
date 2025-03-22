import pytest

from environment_variables import load_env_variables
from pipeline.video_processor import VideoProcessor
from pipeline.face_verifier import FaceVerifierProcess
from pipeline.face_detector import FaceDetector

from pipeline.face_verifier import FaceVerifier

import os
import cv2
import sys
import time
import warnings
import shutil

load_env_variables()

@pytest.fixture
def test_images_dir():
    joined_path = os.path.join(os.path.dirname(__file__), "..", "images/")
    joined_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "images/"))
    print(joined_path)
    return joined_path


@pytest.fixture(scope="module")
def fv_process_instance():
    fv = FaceVerifierProcess()
    yield fv
    fv.stop()

@pytest.fixture(scope="module")
def face_verifier_instance():
    face_verifier = FaceVerifier(
        threshold_stage1=0.2,
        threshold_stage2=0.5,
        debug=False
    )
    return face_verifier
    
@pytest.fixture(scope="module")
def face_detector_instance():
    face_detector = FaceDetector(
        threads=3,
        model_input_shape=[384, 288],
        conf_th=0.7,
        nms_th=0.3,
        topk=5000,
        keep_topk=750,
        debug=False
    )
    return face_detector

@pytest.fixture(scope="module")
def pipeline_instance(fv_process_instance):
    face_store_dir = os.environ["EMBEDDING_STORE_DIRECTORY"]
    print("this is the embedding stoer dir!", face_store_dir)
    try:
        shutil.rmtree(face_store_dir)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))
    return VideoProcessor(fv_process_instance)

   
def test_integration_store_person(pipeline_instance, test_images_dir):
    print("This is the images path", test_images_dir)
    image_path = os.path.join(test_images_dir, "john.png")
    pipeline_instance.store_person(image_path, "John")

    image_path = os.path.join(test_images_dir, "ryan_reynolds.jpg")
    pipeline_instance.store_person(image_path, "Ryan Reynolds")

    image_path = os.path.join(test_images_dir, "tammy_reynolds.jpg")
    pipeline_instance.store_person(image_path, "Tammy Reynolds")
    
    store_query_people = ["John", "Ryan Reynolds", "Tammy Reynolds"]
    
    face_store_dir = os.environ["EMBEDDING_STORE_DIRECTORY"]
    face_store1 = face_store_dir + "stage1_embeddings/"
    face_store2 = face_store_dir + "stage2_embeddings/"
    
    stored_names = []
    missing_names = []
    name_missing = False

    #load embeddings
    for directory in [face_store1, face_store2]:
        for idx, filename in enumerate(os.listdir(directory)):
            f = os.path.join(directory, filename)
            if os.path.isfile(f):
                name = filename.split("_")[1].split(".")[0]
                print(name)
                stored_names.append(name)
                
        for name in store_query_people:
            if name not in stored_names:
                name_missing = True 
                missing_names.append(name)
    
        stored_names = []
        
    assert not name_missing, f"People were not stored correctly. This includes {missing_names}"

def test_integration_pipeline_image1(pipeline_instance, test_images_dir):
    image_path = os.path.join(test_images_dir, "ryan-reynolds-family.jpg")
    frame = cv2.imread(image_path)
    
    track_bbs_ids, obj_track_bbs_ids = pipeline_instance.process_frame(frame)

    num_of_people_det = len(track_bbs_ids)

    # Get results 
    results = []
    while len(results) != num_of_people_det:
        time.sleep(1)
        results.append(pipeline_instance.collect_verification_results())
    
    print("Results from FaceVerifierProcess:")
    found_ryan = False
    found_tammy = True
    for result in results:
        print(result)
        found_ryan_in_result = any(r[1] == 'Ryan Reynolds' for r in result.values())
        found_tammy_in_result = any(r[1] == 'Tammy Reynolds' for r in result.values())
        if found_ryan_in_result:
            found_ryan = True
        if found_tammy_in_result:
            found_tammy = True

    if not found_tammy:
        warnings.warn(UserWarning(f"Tammy Reynolds was not found!"))

    assert found_ryan, "Ryan Reynolds was not found in the image!"
    
def test_integration_face_detector_and_verifier_process1(pipeline_instance, fv_process_instance, face_detector_instance, test_images_dir):
    image_path = os.path.join(test_images_dir, "ryan-reynolds-family.jpg")
    frame = cv2.imread(image_path)
    
    faces_boxes, faces_scores = face_detector_instance.process_frame(frame)
    
    # Submit task
    for i, box in enumerate(faces_boxes):
        print(box)
        fv_process_instance.submit_task(frame, box, task_id=i)
    
    # Collect results
    results = {}
    num_received = 0
    while num_received < len(faces_boxes):
        response = fv_process_instance.get_result(block=True, timeout=5)
        if response is not None:
            task_id, result = response
            if task_id == "ERROR":
                print("Error in worker:", result)
            else:
                results[task_id] = result
                num_received += 1

    
    print(results)
    print("Results from FaceVerifierProcess:")
    found_ryan = any(r[1] == 'Ryan Reynolds' for r in results.values())
    found_tammy = any(r[1] == 'Tammy Reynolds' for r in results.values())

    if not found_tammy:
        warnings.warn(UserWarning(f"Tammy Reynolds was not found!"))

    assert found_ryan, "Ryan Reynolds was not found in the image!"

    
def test_integration_face_detector_and_verifier_process2(pipeline_instance, fv_process_instance, face_detector_instance, test_images_dir):
    image_path = os.path.join(test_images_dir, "ryan-reynolds-family_angle2.jpg")
    frame = cv2.imread(image_path)
    
    faces_boxes, faces_scores = face_detector_instance.process_frame(frame)
    
    for i, box in enumerate(faces_boxes):
        print(box)
        fv_process_instance.submit_task(frame, box, task_id=i)
    
    results = {}
    num_received = 0
    while num_received < len(faces_boxes):
        response = fv_process_instance.get_result(block=True, timeout=5)
        if response is not None:
            task_id, result = response
            if task_id == "ERROR":
                print("Error in worker:", result)
            else:
                results[task_id] = result
                num_received += 1

    print(results)
    print("Results from FaceVerifierProcess:")
    found_ryan = any(r[1] == 'Ryan Reynolds' for r in results.values())
    found_tammy = any(r[1] == 'Tammy Reynolds' for r in results.values())

    if not found_tammy:
        warnings.warn(UserWarning(f"Tammy Reynolds was not found!"))

    assert found_ryan, "Ryan Reynolds was not found in the image!"


def test_integration_face_detector_and_verifier_process3(pipeline_instance, fv_process_instance, face_detector_instance, test_images_dir):
    image_path = os.path.join(test_images_dir, "ryan-reynolds-family_angle3.jpg")
    frame = cv2.imread(image_path)
    
    faces_boxes, faces_scores = face_detector_instance.process_frame(frame)
    
    for i, box in enumerate(faces_boxes):
        print(box)
        fv_process_instance.submit_task(frame, box, task_id=i)
    
    results = {}
    num_received = 0
    while num_received < len(faces_boxes):
        response = fv_process_instance.get_result(block=True, timeout=5)
        if response is not None:
            task_id, result = response
            if task_id == "ERROR":
                print("Error in worker:", result)
            else:
                results[task_id] = result
                num_received += 1

    print(results)
    print("Results from FaceVerifierProcess:")
    found_ryan = any(r[1] == 'Ryan Reynolds' for r in results.values())
    found_tammy = any(r[1] == 'Tammy Reynolds' for r in results.values())

    if not found_tammy:
        warnings.warn(UserWarning(f"Tammy Reynolds was not found!"))

    assert found_ryan, "Ryan Reynolds was not found in the image!"


def test_integration_face_verifier_and_detector(face_detector_instance, face_verifier_instance, test_images_dir):
    image_path = os.path.join(test_images_dir, "ryan-reynolds-family.jpg")
    frame = cv2.imread(image_path)
    
    boxes, scores = face_detector_instance.process_frame(frame)
    
    found_ryan = False
    for box, score in zip(boxes, scores):
        # results = [[int score, str person_name]]
        result = face_verifier_instance.process_face(frame, box)
        if result[1] == "Ryan Reynolds":
            found_ryan = True
    assert found_ryan, "Ryan Reynolds was not found in the image!"
    


    
    