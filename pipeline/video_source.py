from threading import Thread, Lock
import cv2
import time
import av
import io
import os
import queue


'''
VideoStorage Records the video frame by frame and stores the video to disk. It is controlled by VideoStream
'''
class VideoStorage:
    def __init__(self):
        self.output_memory_file = None
        self.output = None
        self.stream = None

    def new_recording(self):
        width, height, fps = 680, 480, 24
        self.output_memory_file = io.BytesIO()
        # Setup in memory file for recording
        self.output = av.open(self.output_memory_file, 'w', format="mp4")
        self.stream = self.output.add_stream("h264", fps)
        # Set encoding parameters for quickest processing
        self.stream.width = width
        self.stream.height = height
        self.stream.pix_fmt = 'yuv444p'
        self.stream.options = {
            'crf': '17',
            'g': str(fps),
            'preset': 'ultrafast',
            'tune': 'zerolatency'
        }

    def submit_frame(self, frame):
        if not self.output or not self.stream:
            return
        # Convert into the correct format
        frame = av.VideoFrame.from_ndarray(frame, format='bgr24')
        packet = self.stream.encode(frame)
        if packet:
            self.output.mux(packet)

    # Save the video, this includes several steps
    def save_video(self, filename):
        if self.output:
            packet = self.stream.encode(None)
            # Encode frame
            if packet:
                self.output.mux(packet)
            self.output.close()

            os.makedirs(os.path.dirname(filename), exist_ok=True)

            # Write to disk
            with open(filename, "wb") as f:
                f.write(self.output_memory_file.getbuffer())
            self.cleanup()
            return filename

    # If the event could not be confirmed the pipeline controller will request to discard the recording
    def discard_recording(self):
        if self.output:
            self.stream.encode(None)
            self.output.close()
        self.cleanup()

    # Reset video storage if recording was saved or discarded
    def cleanup(self):
        self.output = None
        self.stream = None
        if self.output_memory_file:
            self.output_memory_file.close()
            self.output_memory_file = None



class VideoStream:
    # Recorder state constants
    STATE_WAIT = "wait"
    STATE_START_RECORDING = "start_recording"
    STATE_RECORDING = "recording"
    STATE_STOP_RECORDING = "stop_recording"
    STATE_DISCARD_RECORDING = "discard_recording"

    def __init__(self, src):
        # Create clips dir if it not exists
        clips_dir = "saved_clips"
        if not os.path.exists(clips_dir):
            os.makedirs(clips_dir)

        self.src = src
        self.cap = None
        self.grabbed = False
        self.frame = None
        self.running = False
        self.capture_thread = None
        self.is_recording = False

        self.frame_lock = Lock()
        self.frame_queue = queue.Queue(maxsize=10)

        '''
        The record_command_queue is used to asynchronously inform the recording worker thread about start, stop and discrad commants. 
        This decoupling is required as the video record thread cannot be interrupted otherwise video artifacts will occur.
        '''
        self.record_command_queue = queue.Queue()
        self.recording_thread = None
        self.recording_worker_running = False

        self.current_video_path = None
        self.storage = None
        self.recorder_state = self.STATE_WAIT

        self.initialize()

    def initialize(self):
        # Tries to reconnect every 5 seconds
        while not self.running:
            self.cap = cv2.VideoCapture(self.src)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap.set(cv2.CAP_PROP_FPS, 12)
                self.grabbed, self.frame = self.cap.read()
                self.running = True
                self.capture_thread = Thread(target=self.update, daemon=True)
                self.capture_thread.start()
                self.recording_worker_running = True
                self.recording_thread = Thread(target=self.recording_worker, daemon=True)
                self.recording_thread.start()
            else:
                time.sleep(5)

    # Read stream periodically
    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            # When stream ends rerun initialize to search for new stream
            if not ret:
                self.running = False
                self.cap.release()
                self.initialize()
                break
            with self.frame_lock:
                self.grabbed = ret
                self.frame = frame.copy()
            if not self.frame_queue.full():
                self.frame_queue.put_nowait(frame.copy())

    # State manager loop
    def recording_worker(self):
        while self.recording_worker_running:
            # Process recording commands
            try:
                command, filename = self.record_command_queue.get_nowait()
                if command == "start":
                    self.storage = VideoStorage()
                    self.storage.new_recording()
                    self.is_recording = True

                    # Ensures clean start of a recording
                    with self.frame_queue.mutex:
                        self.frame_queue.queue.clear()

                # Store Video
                elif command == "stop" and self.is_recording:
                    self.storage.save_video(filename)
                    self.storage = None
                    self.is_recording = False
                
                # Discard Recording
                elif command == "discard" and self.is_recording:
                    self.storage.discard_recording()
                    self.storage = None
                    self.is_recording = False

                self.record_command_queue.task_done()

            # Triggers if timeout happened
            except queue.Empty:
                pass

            # Process frames if recording is active
            if self.is_recording:
                try:
                    frame = self.frame_queue.get(timeout=0.01)
                    self.storage.submit_frame(frame)
                    self.frame_queue.task_done()
                # Triggers if timeout happened
                except queue.Empty:
                    pass
            # Avoid using all CPU time
            time.sleep(0.001)

    # Called to read most current frame
    def read(self):
        with self.frame_lock:
            return self.grabbed, self.frame

    # Called from pipeline to start recording
    def start_recording(self):
        self.record_command_queue.put(("start", None))

    # Called from pipeline to stop recording
    def stop_recording(self):
        self.record_command_queue.put(("stop", self.current_video_path))

    # Called from pipeline to discard recording
    def discard_recording(self):
        self.record_command_queue.put(("discard", None))

    # Set output file name
    def set_output_filename(self, filename):
        self.current_video_path = filename


# The frame buffer if past between many components as a class, mainly to force pass by reference
class FrameBuffer():
    def __init__(self):
        self._buffer = None
        self.width = 620
        self.height = 480
        
    # If frame buffer is empty because no stream has been read it shows an empty frame that says: 'connecting...'
    def get(self):
        if self._buffer is None:
            image = np.random.randint(0, 256, (self.height, self.width, 3), dtype=np.uint8)
            cv2.putText(image, 'Connecting...', (0, self.height-30), cv2.FONT_HERSHEY_SIMPLEX, 
                        3, (255, 0, 0), 2, cv2.LINE_AA)
            self._buffer = image
        return self._buffer 

    # The set method is overridden by main.py (dynamically)
    def set(self, frame):
        self._buffer = frame