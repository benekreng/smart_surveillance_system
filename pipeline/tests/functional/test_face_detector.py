import pytest

from environment_variables import load_env_variables
import cv2
from pipeline.face_detector import FaceDetector
import os

load_env_variables()

@pytest.fixture
def test_images_dir():
    return os.path.join(os.path.dirname(__file__), "..", "images/")

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

def test_integration_face_detector_1(face_detector_instance, test_images_dir):
    image_path = os.path.join(test_images_dir, "ryan-reynolds-family.jpg")
    frame = cv2.imread(image_path)
    boxes, scores = face_detector_instance.process_frame(frame)
    
    num_of_faces = len(boxes)
    faces_found = False
    
    if num_of_faces > 3:
        faces_found = True
        
    assert faces_found, f"Expected numer of faces where not found. Number of faces found: {num_of_faces}"
    
def test_integration_face_detector_2(face_detector_instance, test_images_dir):
    image_path = os.path.join(test_images_dir, "ryan-reynolds-family_angle2.jpg")
    frame = cv2.imread(image_path)
    boxes, scores = face_detector_instance.process_frame(frame)
    
    num_of_faces = len(boxes)
    faces_found = False
    
    if num_of_faces > 4:
        faces_found = True
        
    assert faces_found, f"Expected numer of faces where not found. Number of faces found: {num_of_faces}"
    
def test_integration_face_detector_3(face_detector_instance, test_images_dir):
    image_path = os.path.join(test_images_dir, "ryan-reynolds-family_angle3.jpg")
    print("IMAGE PATH", image_path) 
    frame = cv2.imread(image_path)
    boxes, scores = face_detector_instance.process_frame(frame)
    
    num_of_faces = len(boxes)
    faces_found = False
    
    if num_of_faces > 4:
        faces_found = True
        
    assert faces_found, f"Excpetced numer of faces where not found. Number of faces found: {num_of_faces}"
        
def test_integration_face_detector_4(face_detector_instance, test_images_dir):
    image_path = os.path.join(test_images_dir, "white.jpg")
    frame = cv2.imread(image_path)
    boxes, scores = face_detector_instance.process_frame(frame)
    
    num_of_faces = len(boxes)
    faces_found = False
    
    if num_of_faces > 0:
        faces_found = True
        
    assert not faces_found, f"Expected numer of faces where not found. Number of faces found: {num_of_faces}"