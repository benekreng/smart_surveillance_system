import copy
import time
from collections import deque
import numpy as np
import cv2
import asyncio
from IPython.display import display, clear_output
from PIL import Image


class FPSCounter:
    # Time interval over which to calculate fps average
    def __init__(self, averaging_interval=1):
        self.averaging_interval = averaging_interval
        self.last_time = time.time()
        self.frame_times = deque()

    def update(self):
        #Update with the current frames timestamp
        current_time = time.time()
        self.frame_times.append(current_time)

        # Remove timestamps outside the window
        while self.frame_times and current_time - self.frame_times[0] > self.averaging_interval:
            self.frame_times.popleft()

    # Calculate and return the average
    def get_fps(self):
        if len(self.frame_times) <= 1:
            return 0.0
        time_span = self.frame_times[-1] - self.frame_times[0]
        # Avoid divide by zero
        return len(self.frame_times) / time_span if time_span > 0 else 0


def draw_detections(frame, body_detections=[], face_detections=[], fps=0, event=False, event_id=-1):
    frame = copy.deepcopy(frame)
    
    def draw_annotations(img, box, tracking_id, class_id):
        # Scale to input image size
        box = np.array(copy.deepcopy(box))
        box[[0, 2]] *= img.shape[1]
        box[[1, 3]] *= img.shape[0]
        box = np.round(box).astype(np.int32)
    
        # Make sure box is in bounds
        x1, y1, x2, y2 = np.clip(box, 0,[img.shape[1]-1, img.shape[0]-1, img.shape[1]-1, img.shape[0]-1])
    
        color = (255, 230, 80)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{class_id} {tracking_id}"
        # Calculate dimensions for label 
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    
        # Calculate the position of the label text
        label_x = x1
        # If y1 is less then 10 avoid negative placement
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10
    
        # Make sure label pos is in bounds
        label_x = max(0, min(label_x, img.shape[1] -label_width))
        label_y = max(label_height, min(label_y, img.shape[0]))
    
        # Draw a filled rectangle as the background for the label text
        cv2.rectangle(
            img, 
            (label_x, label_y -label_height), 
            (label_x + label_width, label_y), 
            color, 
            cv2.FILLED)
    
        # Draw the label text on the image
        cv2.putText(
            img, label, (label_x, label_y), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
            (0, 0, 0), 1, cv2.LINE_AA
        )
        return img
    
    # Draw all body detection bboxes 
    for body_detection in body_detections:
        box = body_detection[0].copy()
        tracking_id = body_detection[1]
        class_id = "Body"
        frame = draw_annotations(frame, box, tracking_id, class_id)
        
    # Draw all face detection bboxes 
    for face_detection in face_detections:
        box = face_detection[0].copy()
        tracking_id = face_detection[1]
        class_id = "Face"
        frame = draw_annotations(frame, box, tracking_id, class_id)


    # Create top label to show fps, event activity, event id and event time
    color = (255, 230, 80)
    fps_label = f"FPS: {int(fps)} Event: {event} ID: {str(event_id)} Time: {time.time()}"
    fps_pos_x = 10
    fps_pos_y = 30
    (label_width, label_height), _ = cv2.getTextSize(fps_label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    
    cv2.rectangle(
        frame, 
        (fps_pos_x, fps_pos_y - label_height), 
        (fps_pos_x + label_width + 5, fps_pos_y + 5), 
        color, 
        cv2.FILLED
    )
    
    cv2.putText(
        frame, fps_label, (fps_pos_x, fps_pos_y), 
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA
    )
    return frame


# Debug class to display framebuffer as video in jupyter lab cell
class JupyterCellDebugView:
    def __init__(self, frame_buffer):
        self.frame_buffer = frame_buffer
        
    # Continuously draw latest frame buffer into jupyter cell
    async def show_frame(self):
        #global output_frame_buffer
        while True:
            try:
                #frame = await frame_queue.get()
                frame_to_display = None
                frame_buffer = self.frame_buffer.get()
                
                if frame_buffer is not None:
                    frame_to_display = frame_buffer
                    clear_output(wait=True)
                    frame_to_display = cv2.cvtColor(frame_to_display, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_to_display)
                    display(pil_image)
                
            except Exception as e:
                print(f"Error displaying frame: {e}")
            # Lock at min 30 fps 
            await asyncio.sleep(0.033)