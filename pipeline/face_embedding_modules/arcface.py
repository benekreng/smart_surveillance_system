import numpy as np
import dlib
import onnx
import onnxruntime as ort
import cv2
from PIL import Image
from torchvision import transforms
import os

'''
Cutting edge feature extractor powering stage 2 of face verification
'''
class ARCFeatureExtractor:
    def __init__(self, debug=False):
        self.debug = debug
        self.predictor = dlib.shape_predictor(os.environ["LANDMARK_PREDICTOR"])
        # Set up onnx runtime
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL 
        self.ort_sess = ort.InferenceSession(os.environ["ARCFACE_MODEL"], sess_options=options, providers=["CPUExecutionProvider"])

    # Main method called by face verifier to obtain face embedding
    def extract(self, img, box):
        box = np.copy(box)
        face = self._pre_process(img, box)
        embedding = self._compute_embedding(face)
        return embedding
        
    def _pre_process(self, img, box):
        '''
        1. Cropps into face
        2. Generates landmarks
        3. Aligns face
        4. Returns aligned face chip
        '''
        image_width, image_height = img.shape[1], img.shape[0]
        box[[0,2]] *= image_width
        box[[1,3]] *= image_height
        box = np.round(box).astype(np.uint)
        
        # box is a x,y,w,h bouning rect
        box = box.astype(int)
        # x, x2, y, y2 (left, right, top, bottom)
        x1, y1, x2, y2 = box
        # (left, top, right, bottom)
        rect_ltrb = [x1, y1, x2, y2]
        pad = 10
        # (left, top, right, bottom) with padding
        ltrb_padded = [x1 - pad, y1 - pad, x2 + pad, y2 + pad]
        # index row first, y first
        cropped_image = img[ltrb_padded[1]:ltrb_padded[3], ltrb_padded[0]:ltrb_padded[2]]
        # signature of rectangle: __init__(self: dlib.rectangle, left: int, top: int, right: int, bottom: int) -> None
        dlib_rectangle = dlib.rectangle(*ltrb_padded)

        if self.debug:
            # debug preview crop
            preview_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
            preview_image.show()

        # convert to grey
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        shape = self.predictor(gray, dlib_rectangle)

        aligned_face = dlib.get_face_chip(img, shape, size=112)
        aligned_face_rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB) 

        if self.debug:
            Image.fromarray(aligned_face_rgb).show()

        return aligned_face_rgb
    
    def _compute_embedding(self, face_chip):
        if self.debug:
            preview_image = Image.fromarray(cv2.cvtColor(face_chip, cv2.COLOR_BGR2RGB))

        img_array = np.array(face_chip).astype(np.float32)
        # Normalize to [-1, 1]
        img_array = (img_array / 127.5) - 1.0
        # Change data layout: from (height, width, channels) to (channels, height, width)
        img_array = np.transpose(img_array, (2, 0, 1))
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        try:
            embedding = self.ort_sess.run(None, {'data': img_array})
        except Exception as e:
            # The arcface model in this repo works with this. However some onnx arcface exports do not work with this
            raise ValueError(f"Depending on the source of the arcface model the key in self.ort_sess.run(None, \
            {'key': img_array}) has to be changed: {e}")
            
        return embedding[0]

    # Calculate Inverse eucledian distance
    def similarity(self, embed_a, embed_b):
        embed_a = np.squeeze(embed_a)
        embed_b = np.squeeze(embed_b)  
        cos_sim = np.dot(embed_a, embed_b) / (np.linalg.norm(embed_a) * np.linalg.norm(embed_b))
        adjusted_sim = (cos_sim + 1) / 2
        return adjusted_sim