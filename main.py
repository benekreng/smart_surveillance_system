import os
import multiprocessing
from multiprocessing import shared_memory
import numpy as np
import cv2
import json
import asyncio
import time
import signal
import sys
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from urllib.parse import unquote
from fastapi import UploadFile, File, Form
from starlette.responses import FileResponse 
import shutil
import tempfile
from environment_variables import load_env_variables
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
import db

# Import pipeline components
from pipeline.face_verifier import FaceVerifierProcess, FaceVerifier
from pipeline.video_source import VideoStream, FrameBuffer
from pipeline_controller import PipelineController
from pipeline.face_detector import FaceDetector
from telegram_bot import TelegramBot

# Load environment variables
load_env_variables()

STREAM_WIDTH = 640
STREAM_HEIGHT = 382
#DEFAULT_RTSP_URL = "rtsp://192.168.2.222:8554/mystream"
DEFAULT_RTSP_URL = "rtsp://192.168.2.217:8554/cam"
EVENT_TIMEOUT = 4
MAX_ANALYSIS_TIME = 10

# Initialize FastAPI app
app = FastAPI(title="Surveillance System")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize components
# Load user preference for bot
preferences = asyncio.run(db.get_user_preferences())
telegram_bot = TelegramBot()
telegram_bot.evaluation_preference(preferences.evaluation_mode)

engine = create_async_engine("sqlite+aiosqlite:///detections.db", echo=False)
async_session = async_sessionmaker(engine, expire_on_commit=False)
face_detector = FaceDetector(
    threads=4,
    model_input_shape=[384, 288],
    conf_th=0.7,
    nms_th=0.3,
    topk=5000,
    keep_topk=750,
    debug=False
)

# Global variables
frame_buffer = None
pipeline_process = None

# Create shared memory on startup
try:
    frame_size = STREAM_WIDTH * STREAM_HEIGHT * 3
    frame_buffer = shared_memory.SharedMemory(name="frame", create=True, size=frame_size)
except Exception as e:
    print(f"Error with shared memory: {e}")


def run_pipeline(rtsp_url, event_timeout, max_analysis_time):
    pipeline_controller = None
    stream = None
    local_frame_buffer = None
    
    try:
        local_frame_buffer = shared_memory.SharedMemory(name="frame")
        frame_buffer = FrameBuffer()
        
        def set_frame_method(frame):
            resized_frame = cv2.resize(frame, (STREAM_WIDTH, STREAM_HEIGHT))
            frame_array = np.ndarray((STREAM_HEIGHT, STREAM_WIDTH, 3), dtype=np.uint8, buffer=local_frame_buffer.buf)
            np.copyto(frame_array, resized_frame)
        # Subsitute set place holder method
        frame_buffer.set = set_frame_method
        
        # Initialize stream and face verifier process
        stream = VideoStream(rtsp_url)
        face_verifier_process = FaceVerifierProcess(threshold_stage1=0.33, threshold_stage2=0.7)
        
        # Start pipeline pipeline_controller
        pipeline_controller = PipelineController(
            stream=stream,
            frame_buffer=frame_buffer,
            face_verifier_process=face_verifier_process,
            face_detector=face_detector,
            db_async_session=async_session,
            telegram_bot=telegram_bot,
            event_timeout=event_timeout,
            max_analysis_time=max_analysis_time
        )
        
        pipeline_controller.running = True
        pipeline_controller.thread() 
    except Exception as e:
        print(f"Pipeline error: {e}")

# Main and only page
@app.get("/")
async def read_index():
    return FileResponse('index.html')

@app.post("/api/start")
async def start_pipeline():
    global pipeline_process
    
    if pipeline_process and pipeline_process.is_alive():
        return{"status": "success", "message": "Pipeline already running"} 
    
    # Start pipeline process
    pipeline_process = multiprocessing.Process(target=run_pipeline, 
                                                args=(DEFAULT_RTSP_URL, EVENT_TIMEOUT, MAX_ANALYSIS_TIME))
    pipeline_process.start()
    
    return {"status": "success", "message":"Pipeline started"}

@app.post("/api/stop")
async def stop_pipeline():
    global pipeline_process
    if not pipeline_process or not pipeline_process.is_alive():
        return {"status": "error", "message": "Pipeline not running"}
    
    pipeline_process.terminate()
    pipeline_process.join(timeout=4)
    
    return {"status": "success", "message": "Pipeline stopped"}

@app.get("/api/status")
async def get_status():
    global pipeline_process
    
    if pipeline_process and pipeline_process.is_alive():
        return {"is_running": True, "state": "running"}
    else:
        return {"is_running": False, "state": "stopped"}

@app.get("/stream/live")
async def stream_live():
    """Stream MJPEG from the live camera"""
    async def generate():
        frame_buffer = None
        try:
            frame_buffer = shared_memory.SharedMemory(name="frame")
            frame_array = np.ndarray((STREAM_HEIGHT, STREAM_WIDTH, 3), 
                                    dtype=np.uint8, buffer=frame_buffer.buf)
            
            while True:
                # Make a copy of the frame from shared memory
                frame = frame_array.copy()
                
                # Create blank frame with message if no valid frame
                if not frame.any():
                    frame = np.zeros((STREAM_HEIGHT, STREAM_WIDTH, 3), dtype=np.uint8)
                    cv2.putText(frame, "Waiting for stream...", (50, STREAM_HEIGHT//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Encode as JPEG and yield
                _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                
                # Control frame rate (about 30 fps)
                await asyncio.sleep(0.033)
        except Exception as e:
            print(f"Stream error: {e}")
        finally:
            if frame_buffer is not None:
                frame_buffer.close()
    
    return StreamingResponse(generate(),media_type="multipart/x-mixed-replace; boundary=frame")

# Send all detections
@app.get("/api/detections")
async def get_detections():
    detections = await db.get_detections(async_session)
    return {"detections": detections}


# stream recorded video as MJPEG
@app.get("/stream/recording/{video_path:path}")
async def stream_recording(request: Request, video_path: str):
    decoded_path = unquote(video_path)
    if not os.path.exists(decoded_path):
        raise HTTPException(status_code=404, detail="Video file not found")
    async def generate():
        cap = cv2.VideoCapture(decoded_path)
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                # Encode frame as jpeg
                _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                
                # Crudely lower fps
                await asyncio.sleep(0.033)
                
        finally:
            cap.release()
    
    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/download/{video_path:path}")
async def download_video(request: Request, video_path: str):
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video file not found")
    
    return FileResponse(path=video_path,filename=os.path.basename(video_path), media_type="video/mp4")

@app.delete("/api/detection/{detection_id}")
async def delete_detection(detection_id: int):
    result = await db.delete_detection(async_session, detection_id)
    if result:
        return {"status": "success", "message": "Detection deleted"}
    else:
        raise HTTPException(status_code=404, detail="Detection not found")

@app.delete("/api/detections/all")
async def delete_all_detections():
    result = await db.delete_all_detections(async_session)
    return { "status": "success", "message": "Deleted all detections"}

#Add a new identity with user-provided name
@app.post("/api/identities/add")
async def add_identity(name: str = Form(...), image: UploadFile = File(...)):
    try:
        # Save the uploaded file temporarily
        temp_dir = tempfile.mkdtemp()
        temp_path = f"{temp_dir}/{image.filename}"
        
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        frame = cv2.imread(temp_path)
        if frame is None:
            return {"status": "error", "message": "Image invalid. Try another image"}

        # Detect face
        face_boxes, _ = face_detector.process_frame(frame)

        if len(face_boxes) == 0:
            return {"status": "error", "message": "No face detected. Try another image"}

        face_verifier = FaceVerifier()
        # Store the first detected face
        box = face_boxes[0]
        result = face_verifier.store_person(frame, box, name)
        del face_verifier

        # Remove temporary files
        os.remove(temp_path)
        os.rmdir(temp_dir)

        return {"status": "success", "message": f"Identity '{name}' added", "result": result}

    except Exception as e:
        return {"status": "error", "message": str(e)}


# Get stored identities from stored embeddings
@app.get("/api/identities")
async def list_identities():
    """Retrieve a list of stored identities from the embeddings directory"""
    try:
        identities = []
        face_store_dir = os.environ["EMBEDDING_STORE_DIRECTORY"]

        for filename in os.listdir(face_store_dir+"stage1_embeddings/"):
            if filename.endswith(".npy"):
                index, name = filename.split("_")[0], filename.split("_")[1].split(".")[0]
                identities.append({"index": index, "name": name})

        return {"status": "success", "identities": identities}

    except Exception as e:
        return {"status": "error", "message": str(e)}


# Delete identities/stored embeddings
@app.delete("/api/identities/delete/{identity_name}")
async def delete_identity(identity_name: str):
    try:
        face_store_dir = os.environ["EMBEDDING_STORE_DIRECTORY"]
        stage1_dir = face_store_dir + "stage1_embeddings/"
        stage2_dir = face_store_dir + "stage2_embeddings/"

        # find file in stage 1 emeddings
        stage1_files_to_remove = [
            os.path.join(stage1_dir, f)
            for f in os.listdir(stage1_dir)
            if f.endswith(".npy") and identity_name in f
        ]
        # find file in stage 2 emeddings
        stage2_files_to_remove = [
            os.path.join(stage2_dir, f)
            for f in os.listdir(stage2_dir)
            if f.endswith(".npy") and identity_name in f
        ]

        if not stage1_files_to_remove and not stage2_files_to_remove:
            raise ValueError("Identity not found")

        # Delete the files
        for file_path in stage1_files_to_remove:
            os.remove(file_path)
        for file_path in stage2_files_to_remove:
            os.remove(file_path)

        return {"status": "success", "message": f"Identity '{identity_name}' removed"}

    except Exception as e:
        return {"status": "error", "message": str(e)}

# Get user preferences
@app.get("/api/preferences")
async def get_user_preferences():
    preferences = await db.get_user_preferences()
    return {"preferences": preferences}

# Evaluate preferences
@app.post("/api/settings/set-evaluation-preference")
async def set_evaluation_preference(data: dict):
    eval_preference = data.get("evaluation_preference")
    await db.set_user_preferences(evaluation_mode=eval_preference)
    
    try:
        telegram_bot.evaluation_preference(eval_preference)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error setting mode to {eval_preference}: {str(e)}")


def cleanup():
    global pipeline_process, frame_buffer
    # Stop pipeline
    if pipeline_process and pipeline_process.is_alive():
        try:
            pipeline_process.terminate()
            pipeline_process.join(timeout=2)
        except Exception as e:
            print(f"Error stopping pipeline: {e}")
    
    # Delete shared memory
    if frame_buffer:
        try:
            frame_buffer.close()
            frame_buffer.unlink()
        except Exception as e:
            print(f"Error closing shared memory: {e}")
    
    print("Cleanup complete")

@app.on_event("shutdown")
async def shutdown_event():
    cleanup()

if __name__ == "__main__":
    import uvicorn
    
    # Very important: Using fork will crash the application for some unkown reason after an event finished evaluating 
    multiprocessing.set_start_method('spawn', force=True)

    # Uvicorn handles signals (ctrl + c) and triggers
    config = uvicorn.Config(app, host="0.0.0.0", port=8000)
    server = uvicorn.Server(config)
    
    try:
        server.run()
    finally:
        cleanup()