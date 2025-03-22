import cv2
import numpy as np
import ncnn
from PIL import Image
import json
import os


'''
EXTERNAL CODE USE: 
Parts of this class were copied and heavily inspired from here:
from here: https://github.com/Tencent/ncnn/blob/master/python/ncnn/model_zoo/yolov8.py
'''

class BodyDetection:
    def __init__(self, threads, model_input_shape, confidence_thres, iou_thres, debug=False):
        self.debug = debug
        # Load the class names from the COCO dataset
        with open(os.environ["COCO8_CLASSES"], 'r') as f:
            self.classes = json.load(f)
            f.close()
        # Generate a color palette for the classes
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))
        
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres

        self.reg_max = 16
        self.mean_vals = []
        self.norm_vals = [1 / 255.0, 1 / 255.0, 1 / 255.0]
        
        self.net = ncnn.Net()
        self.net.opt.num_threads=threads
        self.net.load_param(os.environ["IMAGE_DETECTION_PARAM"])
        self.net.load_model(os.environ["IMAGE_DETECTION_BIN"]) 
        self.net.opt.use_winograd_convolution = True
        ncnn.set_cpu_powersave(2)
        
        self.target_size_x = model_input_shape[0]
        self.target_size_y = model_input_shape[1]
        
        self.net.opt.use_fp16_packed = True
        self.net.opt.use_fp16_storage = True
        self.net.opt.use_fp16_arithmetic = True

    def detect(self, image):
        image_data = self.preprocess(image)
        # Run inference 
        ex = self.net.create_extractor()
        ex.input("in0", image_data)
        ret, mat_out = ex.extract("out0")

        boxes, scores = self.postprocess(image, mat_out)
        return boxes, scores
    
    def preprocess(self, img):
        # Read the input image using OpenCV
        if img is None:
            raise ValueError(f"Error: Unable to load image from path {image_path}")
    
        self.img_w, self.img_h = img.shape[1], img.shape[0]
    
        # Scale factors for width and height
        scale_x = self.target_size_x / self.img_w
        scale_y = self.target_size_y / self.img_h
        scale = min(scale_x, scale_y)
    
        # Compute new scaled dimensions
        new_w = int(self.img_w * scale)
        new_h = int(self.img_h * scale)
    
        # Resize the image
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
        # Compute padding for different target resolutions
        wpad = self.target_size_x - new_w
        hpad = self.target_size_y - new_h
    
        # Ensure padding is non-negative
        wpad = max(0, wpad)
        hpad = max(0, hpad)
    
        # Convert to ncnn format
        mat_in = ncnn.Mat.from_pixels_resize(
            resized_img, ncnn.Mat.PixelType.PIXEL_BGR2RGB, new_w, new_h, new_w, new_h
        )
    
        # Apply padding
        mat_in_pad = ncnn.copy_make_border(
            mat_in,
            hpad // 2,
            hpad - hpad // 2,
            wpad // 2,
            wpad - wpad // 2,
            ncnn.BorderType.BORDER_CONSTANT,
            114.0,
        )
    
        self.wpad = wpad
        self.hpad = hpad
        self.scaled_width = new_w
        self.scaled_height = new_h
    
        # Normalize
        mat_in_pad.substract_mean_normalize(self.mean_vals, self.norm_vals)
    
        # Convert to NumPy array for debugging
        h, w, c = mat_in_pad.h, mat_in_pad.w, mat_in_pad.c
        data = np.zeros((h, w, c), dtype=np.float32)
    
        for i in range(c):
            channel_data = np.array(mat_in_pad.channel(i), dtype=np.float32)
            data[:, :, i] = channel_data.reshape((h, w))
    
        # Convert to uint8 for visualization
        data = np.clip(data * 255, 0, 255).astype(np.uint8)
        data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
    
        # Return the preprocessed image
        return mat_in_pad

    def postprocess(self, input_image, output):
        """
        Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

        Args:
            input_image (numpy.ndarray): The input image.
            output (numpy.ndarray): The output of the model.

        Returns:
            numpy.ndarray: The input image with detections drawn on it.
        """
        # Transpose and squeeze the output to match the expected shape
        outputs = np.transpose(np.squeeze(output))
        # Get the number of rows in the outputs array
        rows = outputs.shape

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        class_ids = []

        orig_width = self.target_size_x - self.wpad
        orig_height = self.target_size_y - self.hpad
        x_factor = self.img_w / self.scaled_width
        y_factor = self.img_h / self.scaled_height

        # Code modify to only support selected class (person), due to slighly increased performance
        # Iterate over each row in the outputs array
        # Cut array to only include boudning boxes and person scores
        outputs = outputs[:,:5]
        
        for i in range(rows[0]):
            # Extract the class scores from the current row
            classes_scores = outputs[i][:5]

            # If the maximum score is above threshold
            if (max_score := outputs[i][4]) >= self.confidence_thres:
                # Extract the bounding box coordinates from the current row
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # Shift to the begginng of the padding, ignoring the padding
                x_padding = self.wpad/2
                x -= x_padding
                y_padding = self.hpad/2
                y -= y_padding

                # Convert from x_center, y_center, w, h to x1, y1, x2, y2
                if self.target_size_y > self.target_size_x:
                    left = int((x - w / 2))
                    width = int(w)
                    top = int((y - h / 2))
                    height = int(h)
                else:
                    left = int((x - w / 2))
                    width = int(w)
                    top = int((y - h / 2))
                    height = int(h)


                x1, y1, x2, y2 = left, top, left + width, top + height

                # Add the class ID, score, and box coordinates to the respective lists
                #box = np.array([left, top, width, height]).astype(np.float64)
                box = np.array([x1, y1, x2 , y2]).astype(np.float64)
                class_ids.append(0)
                scores.append(max_score)
                
                box[[0,2]] /= self.scaled_width
                box[[1,3]] /= self.scaled_height
                boxes.append(box)

        
        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)

        boxes = np.array(boxes)[indices]
        scores = np.array(scores)[indices]
        class_ids = np.array(class_ids)[indices]

        # "Np hstack" boxes
        if len(boxes) ==  0:
            return np.empty((0, 4)), np.empty((0, 1))
            
        array1 = boxes
        array2 = scores
        # fused_array = np.hstack((array1, array2.reshape(-1, 1)))
        # Return the modified input image
        return boxes, scores
        