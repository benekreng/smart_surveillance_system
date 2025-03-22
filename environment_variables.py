import os

def load_env_variables():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    os.environ["EMBEDDING_STORE_DIRECTORY"] = os.path.join(BASE_DIR, "embedding_store/")
    os.environ["COCO8_CLASSES"] = os.path.join(BASE_DIR, "models/coco8_classes.json")
    os.environ["LANDMARK_PREDICTOR"] = os.path.join(BASE_DIR, "models/shape_predictor_68_face_landmarks.dat")
    os.environ["IMAGE_DETECTION_PARAM"] = os.path.join(BASE_DIR, "models/yolo11n_ncnn_model160x160/model.ncnn.param")
    os.environ["IMAGE_DETECTION_BIN"] = os.path.join(BASE_DIR, "models/yolo11n_ncnn_model160x160/model.ncnn.bin")
    os.environ["FACE_DETECTOR_MODEL"] = os.path.join(BASE_DIR, "models/face_detection_yunet.onnx")
    os.environ["EDGEFACE_MODEL"] = os.path.join(BASE_DIR, "models/edgeface_s_gamma_05.onnx")
    os.environ["ARCFACE_MODEL"] = os.path.join(BASE_DIR, "models/arcface.onnx")

    