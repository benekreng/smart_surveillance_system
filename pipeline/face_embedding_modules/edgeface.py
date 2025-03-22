import numpy as np
import dlib
import onnx
import onnxruntime as ort
import cv2
from PIL import Image
import torch
from torchvision import transforms
import os

'''
Feature extractor used for stage 1 of the face verifier. It is not used as the main embedding extractor due to its low performance.
It is employed in stage 1 of the face verifier to filter out low quality faces to avoid spending much execution time
on computing Arc Face embeddings of bad detections. (Inference time is 50ms compared to 300-400ms of arcface)
'''

class EFFeatureExtractor:
    def __init__(self, debug=False):
        self.debug = debug
        self.predictor = dlib.shape_predictor(os.environ["LANDMARK_PREDICTOR"])
        # Setup onnx runtime
        self.onnx_model = onnx.load(os.environ["EDGEFACE_MODEL"])
        onnx.checker.check_model(self.onnx_model)
        # Limit parallelism
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL 

        self.ort_sess = ort.InferenceSession(os.environ["EDGEFACE_MODEL"], sess_options=options, providers=["CPUExecutionProvider"])

    # Called by pipeline controller to extract face embeddings
    def extract(self, img, box):
        box = np.copy(box)
        face = self._pre_process(img, box)
        embedding = self._compute_embedding(face)
        return embedding
        
    def _pre_process(self, img, box):
        '''
        cropps into face, generates landmarks, aligns face
        returns aligned face chip
        '''
        image_width, image_height = img.shape[1], img.shape[0]
        box[[0,2]] *= image_width
        box[[1,3]] *= image_height
        box = np.round(box).astype(np.uint)
        
        # box is a x,y,w,h bouning rect
        box = box.astype(int)
        #x, x2, y, y2 (left, right, top, bottom)
        x1, y1, x2, y2 = box
        #(left, top, right, bottom)
        rect_ltrb = [x1, y1, x2, y2]
        pad = 10
        #(left, top, right, bottom) with padding
        ltrb_padded = [x1 - pad, y1 - pad, x2 + pad, y2 + pad]
        #index row first, y first
        cropped_image = img[ltrb_padded[1]:ltrb_padded[3], ltrb_padded[0]:ltrb_padded[2]]
        # signature of rectangle: __init__(self: dlib.rectangle, left: int, top: int, right: int, bottom: int) -> None
        dlib_rectangle = dlib.rectangle(*ltrb_padded)
    
        if self.debug:
            # Debug preview crop
            preview_image = Image.fromarray(cv.cvtColor(cropped_image, cv.COLOR_BGR2RGB))
            preview_image.show()
    
        # Convert to grey
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        shape = self.predictor(gray, dlib_rectangle)
        
        aligned_face = dlib.get_face_chip(img, shape, size=112)
        
        if self.debug:
            Image.fromarray(aligned_face).show()

        return aligned_face
    
    def _compute_embedding(self, face_chip):
        if self.debug:
            preview_image = Image.fromarray(cv.cvtColor(face_chip, cv.COLOR_BGR2RGB))
        # Convert to pytorch tensor and normalize
        face_chip = torch.tensor(face_chip, dtype=torch.float32).permute(2, 0, 1) / 255.0
        # Normalize
        transform = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        transformed_input = transform(face_chip) 
        transformed_input = transformed_input.unsqueeze(0)
        # Predict
        embedding = self.ort_sess.run(None, {'input': np.array(transformed_input)})
        return embedding[0]

    # Calculate normalized eucledian distance
    def similarity(self, embed_a, embed_b):
        embed_a = embed_a / np.linalg.norm(embed_a)
        embed_b = embed_b / np.linalg.norm(embed_b)
        euclidean_distance = np.linalg.norm(embed_a - embed_b)
        # Scale to range 0, 1
        similarity_score = 1 - (euclidean_distance / 2)
        return similarity_score