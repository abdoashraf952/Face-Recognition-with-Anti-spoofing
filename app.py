import os
import cv2
import threading
from ultralytics import YOLO
from simple_facerec import SimpleFacerec
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, UploadFile, File
import requests
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# General settings
confidence_threshold = 0.6
save_interval = timedelta(seconds=10)
BACKEND_URL = "http://192.168.1.32:5116/api/Images/receive-from-ai"

# Load YOLO model
model = YOLO("best.pt")
classNames = ["fake", "real"]

# Load face recognition model
sfr = SimpleFacerec()
sfr.load_encoding_images("data/")

# Variables to track camera state and saved faces
camera_running = False
cap = None
saved_faces = set()
last_seen_faces = {}
saved_fake = False
last_seen_fake_time = None
prev_frame_time = 0

def process_camera():
    global saved_fake, last_seen_fake_time, saved_faces, last_seen_faces, prev_frame_time

    # Open webcam
    cap.set(3, 640)
    cap.set(4, 480)

    while camera_running:
        success, frame = cap.read()
        current_time = datetime.now()

        # Run YOLO to detect real/fake faces
        is_real_detected = False
        is_fake_detected = False
        results = model(frame, stream=True, verbose=False)

        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                if conf > confidence_threshold:
                    label = classNames[cls]

                    if label == 'real':
                        is_real_detected = True
                    elif label == 'fake':
                        is_fake_detected = True
                        last_seen_fake_time = current_time

                    
        # Clear fake flag if enough time has passed since last fake detection
        if saved_fake and last_seen_fake_time:
            if (current_time - last_seen_fake_time) > save_interval:
                saved_fake = False

        # If FAKE is detected, skip face recognition and save image once
        if is_fake_detected:
            if not saved_fake:
                file_name = f"fake_{current_time.strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
                cv2.imwrite(file_name, frame.copy())

                # Upload the image to backend
                upload_image(file_name,'not recognized', "fake", current_time)

                saved_fake = True

        # If no FAKE detected, proceed with face recognition
        else:
            _, face_names = sfr.detect_known_faces(frame)

            # Update last seen time for each face
            for name in face_names:
                last_seen_faces[name] = current_time

            # Remove faces from saved set if they haven't been seen recently
            to_remove = [name for name in saved_faces if name not in face_names and
                         (current_time - last_seen_faces.get(name, current_time)) > save_interval]
            for name in to_remove:
                saved_faces.remove(name)

            # Save frame if a new known/unknown face is detected along with "real" classification
            name_to_save = None
            for name in face_names:
                if name not in saved_faces and is_real_detected:
                    name_to_save = name if name != "Unknown" else "Unknown"
                    break

            if name_to_save:
                file_name = f"{name_to_save}_{current_time.strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
                cv2.imwrite(file_name, frame.copy())

                # Upload the image to backend
                upload_image(file_name, name_to_save,'real', current_time)

                saved_faces.add(name_to_save)


    cap.release()
    cv2.destroyAllWindows()

def upload_image(file_name, name ,classification, current_time):
    with open(file_name, "rb") as f:
        try:
            response = requests.post(
                BACKEND_URL,
                files={"file": f},
                data={
                    "name":name ,
                    "classification": classification,
                    "timestamp": current_time.replace(microsecond=0).isoformat(),
                },
            )
            print(f"[SUCCESS] Frame sent: {response.status_code}, {response.text}")
        except Exception as e:
            print(f"[ERROR] Failed to send frame: {e}")

@app.post("/start-camera")
def start_camera():
    global camera_running, cap

    if camera_running:
        raise HTTPException(status_code=400, detail="Camera is already running.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Failed to access the camera.")

    camera_running = True
    threading.Thread(target=process_camera, daemon=True).start()
    return {"status": "Camera started"}

@app.post("/upload-face")
def upload_face_image(file: UploadFile = File(...)):
    save_path = os.path.join("data", file.filename)

    with open(save_path, "wb") as buffer:
        buffer.write(file.file.read())

    sfr.load_encoding_images("data/")

    return {"status": f"Face image '{file.filename}' uploaded and encodings updated."}

@app.post("/stop-camera")
def stop_camera():
    global camera_running, cap

    if not camera_running:
        raise HTTPException(status_code=400, detail="Camera is not running.")

    camera_running = False
    if cap:
        cap.release()
    cv2.destroyAllWindows()

    return {"status": "Camera stopped"}
# uvicorn app:app --reload
#uvicorn app:app --host 0.0.0.0 --port 8000 --reload