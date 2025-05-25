import math
import time
import cv2
import cvzone
from ultralytics import YOLO
from simple_facerec import SimpleFacerec
from datetime import datetime, timedelta

# General settings
confidence_threshold = 0.6
save_interval = timedelta(seconds=10)

# Load YOLO model
model = YOLO("best.pt")
classNames = ["fake", "real"]

# Load face recognition model
sfr = SimpleFacerec()
sfr.load_encoding_images("data/")

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Tracking saved faces to avoid duplicates
saved_faces = set()
last_seen_faces = {}

# Tracking fake detections
saved_fake = False
last_seen_fake_time = None

# FPS tracking
prev_frame_time = 0

while True:
    new_frame_time = time.time()
    success, frame = cap.read()
    current_time = datetime.now()

    # Run YOLO to detect real/fake faces
    is_real_detected = False
    is_fake_detected = False
    results = model(frame, stream=True, verbose=False)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            if conf > confidence_threshold:
                label = classNames[cls]
                color = (0, 255, 0) if label == 'real' else (0, 0, 255)

                if label == 'real':
                    is_real_detected = True
                elif label == 'fake':
                    is_fake_detected = True
                    last_seen_fake_time = current_time

                # Draw bounding box and label
                cvzone.cornerRect(frame, (x1, y1, x2 - x1, y2 - y1), colorC=color, colorR=color)
                cvzone.putTextRect(frame, f'{label.upper()} {int(conf*100)}%',
                                   (max(0, x1), max(35, y1)), scale=2, thickness=4,
                                   colorR=color, colorB=color)

    # Clear fake flag if enough time has passed since last fake detection
    if saved_fake and last_seen_fake_time:
        if (current_time - last_seen_fake_time) > save_interval:
            saved_fake = False

    # === If FAKE is detected, skip face recognition and save image once
    if is_fake_detected:
        if not saved_fake:
            file_name = f"fake_{current_time.strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
            cv2.imwrite(file_name, frame.copy())
            print(f"[Saved FAKE] {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            saved_fake = True

    # === If no FAKE detected, proceed with face recognition
    else:
        face_locations, face_names = sfr.detect_known_faces(frame)

        # Update last seen time for each face
        for name in face_names:
            last_seen_faces[name] = current_time

        # Remove faces from saved set if they haven't been seen recently
        to_remove = [name for name in saved_faces if name not in face_names and
                     (current_time - last_seen_faces.get(name, current_time)) > save_interval]
        for name in to_remove:
            saved_faces.remove(name)

        # Draw face bounding boxes and names
        for face_loc, name in zip(face_locations, face_names):
            y1, x2, y2, x1 = face_loc
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)

        # Save frame if a new known/unknown face is detected along with "real" classification
        name_to_save = None
        for name in face_names:
            if name not in saved_faces and is_real_detected:
                name_to_save = name if name != "Unknown" else "Unknown"
                break

        if name_to_save:
            file_name = f"{name_to_save}_{current_time.strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
            cv2.imwrite(file_name, frame.copy())
            print(f"[Saved REAL] {current_time.strftime('%Y-%m-%d %H:%M:%S')} - {name_to_save}")
            saved_faces.add(name_to_save)

    # Display frame with FPS
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    cv2.imshow("Image", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
