import pytest

from environment_variables import load_env_variables
import cv2
from pipeline.body_detector import BodyDetection

import numpy as np
import os

load_env_variables()

@pytest.fixture
def test_images_dir():
    return os.path.join(os.path.dirname(__file__), "..", "images/")

@pytest.fixture(scope="module")
def obj_detector_instance():
    obj_detector = BodyDetection(
        threads=2,
        model_input_shape=[320, 320],
        confidence_thres=0.60,
        iou_thres=0.45,
        debug=False
    )
    return obj_detector

def test_integration_people_detection1(obj_detector_instance, test_images_dir):
    image_path = os.path.join(test_images_dir, "bus.jpg")
    image = cv2.imread(image_path)

    bboxes, scores = obj_detector_instance.detect(image)

    people_detected = False
    num_of_detections = len(bboxes)
    if num_of_detections > 2:
        people_detected = True

    assert people_detected, f"less than 3 people detected: {num_of_detections}"
    
def test_integration_people_detection2(obj_detector_instance, test_images_dir):
    image_path = os.path.join(test_images_dir, "ryan-reynolds-family.jpg")
    image = cv2.imread(image_path)

    bboxes, scores = obj_detector_instance.detect(image)

    people_detected = False
    num_of_detections = len(bboxes)
    if num_of_detections > 3:
        people_detected = True

    assert people_detected, f"less than 3 people detected: {num_of_detections}"


def test_integration_people_detection3(obj_detector_instance, test_images_dir):
    image_path = os.path.join(test_images_dir, "ryan-reynolds-family_angle2.jpg")
    image = cv2.imread(image_path)

    bboxes, scores = obj_detector_instance.detect(image)

    people_detected = False
    num_of_detections = len(bboxes)
    if num_of_detections > 3:
        people_detected = True

    assert people_detected, f"less than 3 people detected: {num_of_detections}"