import copy
from itertools import product
import onnxruntime
import math
from PIL import Image
import copy
import time
import cv2 as cv
import numpy as np
import random
import os


'''
EXTERNAL CODE USE: 
Parts of this class were copied and heavily inspired from here:
from here: https://github.com/Kazuhito00/YuNet-ONNX-TFLite-Sample/blob/main/yunet/yunet_onnx.py
'''

class FaceDetector(object):

    # Feature map
    MIN_SIZES = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
    STEPS = [8, 16, 32, 64]
    VARIANCE = [0.1, 0.2]

    def __init__(
        self,
        threads,
        model_input_shape,
        conf_th=0.6,
        nms_th=0.3,
        topk=5000,
        keep_topk=750,
        debug=False
    ):
        options = onnxruntime.SessionOptions()
        # Limit parallelism
        options.intra_op_num_threads = threads  
        options.inter_op_num_threads = threads
        options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL 

        self.onnx_session = onnxruntime.InferenceSession(os.environ["FACE_DETECTOR_MODEL"], sess_options=options, providers=['CPUExecutionProvider'])
        self.onnx_session.set_providers(['CPUExecutionProvider'], [{'intra_op_num_threads': threads, 'precision': 'fp16'}])

        self.input_name = self.onnx_session.get_inputs()[0].name
        output_name_01 = self.onnx_session.get_outputs()[0].name
        output_name_02 = self.onnx_session.get_outputs()[1].name
        output_name_03 = self.onnx_session.get_outputs()[2].name
        self.output_names = [output_name_01, output_name_02, output_name_03]

        self.model_input_shape = model_input_shape
        self.conf_th = conf_th
        self.nms_th = nms_th
        self.topk = topk
        self.keep_topk = keep_topk

        self.priors = None
        self.debug = debug
        self._generate_priors()

    def _generate_priors(self):
        w, h = self.model_input_shape

        feature_map_2th = [
            int(int((h + 1) / 2) / 2),
            int(int((w + 1) / 2) / 2)
        ]
        feature_map_3th = [
            int(feature_map_2th[0] / 2),
            int(feature_map_2th[1] / 2)
        ]
        feature_map_4th = [
            int(feature_map_3th[0] / 2),
            int(feature_map_3th[1] / 2)
        ]
        feature_map_5th = [
            int(feature_map_4th[0] / 2),
            int(feature_map_4th[1] / 2)
        ]
        feature_map_6th = [
            int(feature_map_5th[0] / 2),
            int(feature_map_5th[1] / 2)
        ]

        feature_maps = [
            feature_map_3th, feature_map_4th, feature_map_5th, feature_map_6th
        ]

        priors = []
        for k, f in enumerate(feature_maps):
            min_sizes = self.MIN_SIZES[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / w
                    s_ky = min_size / h

                    cx = (j + 0.5) * self.STEPS[k] / w
                    cy = (i + 0.5) * self.STEPS[k] / h

                    priors.append([cx, cy, s_kx, s_ky])

        self.priors = np.array(priors, dtype=np.float32)

    # Load image, process, and display
    def process_frame(self, original_image):
        if original_image is None:
            print("Error: Image not found!")
            return
    
        bboxes, scores = self.inference(original_image)
    
        
        if self.debug:
            debug_image = self._draw_debug(original_image, bboxes, scores)
            # Draw results
            preview_image = Image.fromarray(cv.cvtColor(debug_image, cv.COLOR_BGR2RGB))
            preview_image.show()
    
        return bboxes, scores
        
    def inference(self, image, return_results=False):
        temp_image = self._preprocess(image)

        result = self.onnx_session.run(
            self.output_names,
            {self.input_name: temp_image},
        )
        
        dets = self._postprocess(result)

        # If dets holds no boxes return empty
        if len(dets[0]) == 0:
            return np.empty((0, 4)), np.empty((0, 1))
        bboxes = dets[:,:4]
        scores = dets[:,4]
        return bboxes, scores

    def _preprocess(self, image):
        # Get the height and width of the input image
        img_h, img_w = image.shape[:2]
        
        # Convert the image color space from BGR to RGB
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        
        nn_input_h = self.model_input_shape[1]
        nn_input_w = self.model_input_shape[0]
        
        # Resize the image first to match the model input size while maintaining aspect ratio
        scale = min(nn_input_w / img_w, nn_input_h / img_h)
        resized_w = int(img_w * scale)
        resized_h = int(img_h * scale)
        
        image = cv.resize(image, (resized_w, resized_h), interpolation=cv.INTER_LINEAR)
        
        # Calculate padding to fit the exact model input size
        top_padding = (nn_input_h - resized_h) // 2
        bottom_padding = nn_input_h - resized_h - top_padding
        left_padding = (nn_input_w - resized_w) // 2
        right_padding = nn_input_w - resized_w - left_padding
        
        self.orig_width = resized_w
        self.orig_height = resized_h
        
        self.hpad = top_padding + bottom_padding
        self.wpad = left_padding + right_padding
        
        # Apply padding to the resized image
        padded = cv.copyMakeBorder(image,
                                   top_padding,
                                   bottom_padding,
                                   left_padding,
                                   right_padding,
                                   cv.BORDER_CONSTANT)
        
        image = padded.astype(np.float32)
        image = image.transpose((2, 0, 1))
        image = image.reshape(1, 3, self.model_input_shape[1], self.model_input_shape[0])
        
        return image 

    
    def _postprocess(self, result):
        dets = self._decode(result)

        # NMS
        keepIdx = cv.dnn.NMSBoxes(
            bboxes=dets[:, 0:4].tolist(),
            scores=dets[:, -1].tolist(),
            score_threshold=self.conf_th,
            nms_threshold=self.nms_th,
            top_k=self.topk,
        )

        # Bboxes, landmarks, scores
        if len(keepIdx) == 0:
            return [], []
            
        dets = dets[keepIdx.flatten()]
        if len(dets.shape) == 3:
            dets = np.squeeze(dets, axis=1)
            
        # Shift to the begginng of the padding, essentially removing the padding (padding is always evently split)
        # Boxes are in dets[:,[0,1,2,3]] (dets[:,:4])
        x_padding = self.wpad//2
        y_padding = self.hpad//2
        dets[:,0] -= x_padding
        dets[:,1] -= y_padding

        # normalising coordinates for output
        #(using numpy fancing indexing)
        # model_input_shape = model output shape
        dets[:,[0,2]] /= self.orig_width
        dets[:,[1,3]] /= self.orig_height

        # convert from x1, y1, w, h -> x1, y1, x2, y2
        dets[:,[2, 3]] = dets[:,[0, 1]] + dets[:,[2, 3]]
        return dets

    def _decode(self, result):
        loc, conf, iou = result

        cls_scores = conf[:, 1]
        iou_scores = iou[:, 0]

        _idx = np.where(iou_scores < 0.)
        iou_scores[_idx] = 0.
        _idx = np.where(iou_scores > 1.)
        iou_scores[_idx] = 1.
        scores = np.sqrt(cls_scores * iou_scores)
        scores = scores[:, np.newaxis]

        scale = np.array(self.model_input_shape)

        bboxes = np.hstack(
            ((self.priors[:, 0:2] +
              loc[:, 0:2] * self.VARIANCE[0] * self.priors[:, 2:4]) * scale,
             (self.priors[:, 2:4] * np.exp(loc[:, 2:4] * self.VARIANCE)) *
             scale))
        bboxes[:, 0:2] -= bboxes[:, 2:4] / 2

        dets = np.hstack((bboxes, scores))

        return dets

    # Debug function to draw detection results on the image
    def _draw_debug(self, image, bboxes, scores):
        image_width, image_height = image.shape[1], image.shape[0]
        debug_image = copy.deepcopy(image)
        bboxes_copy = np.copy(bboxes)
        for bbox, score in zip(bboxes_copy, scores):
            bbox[[0,2]] *= image.shape[1]
            bbox[[1,3]] *= image.shape[0]
            bbox = np.round(bbox).astype(np.uint)
            x1, y1, w, h  = bbox
    
            color = (random.randint(1, 255), random.randint(1, 255), random.randint(1, 255))
            
            x1, y1, x2, y2 = bbox
            cv.rectangle(debug_image, (x1, y1), (x2, y2), color, 2)
            cv.putText(debug_image, f'{score:.4f}', (x1, y1 - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return debug_image
    
