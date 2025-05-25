# Face-Recognition-with-Anti-spoofing
This FastAPI-based project uses YOLOv11 and face recognition to classify real vs. fake faces from a live webcam stream. It also detects known/unknown persons and uploads captured frames to a backend server with metadata. Real-time image classification and facial recognition are combined into an efficient and automated surveillance solution.


# AI Real/Fake Face Detection and Recognition System

This is a real-time AI-powered surveillance system built with **FastAPI**, **YOLOv8**, and **face recognition**. The system captures live video, detects whether a face is **real or fake**, and performs face recognition. Detected frames are automatically saved and sent to a backend API for storage.

---

## 🚀 Features

- 🔍 **Real-time YOLOv8 face classification**: Detects whether a face is "real" or "fake".
- 🧠 **Face recognition**: Identifies known individuals using encodings from a dataset.
- 📤 **Automatic image upload**: Detected faces (real or fake) are saved and sent to a backend endpoint.
- ⚡ **FastAPI-based API**: Includes endpoints to start/stop the camera and upload new face images for training.
- 🔁 **Smart saving logic**: Avoids saving duplicate frames within a defined interval.

---

## 🛠️ Tech Stack

- [FastAPI](https://fastapi.tiangolo.com/) - Python Web Framework
- [Ultralytics YOLOv11](https://docs.ultralytics.com/) - Real/Fake Face Detection
- [OpenCV](https://opencv.org/) - Video capture and image processing
- [SimpleFacerec](https://github.com/ageitgey/face_recognition) - Face recognition utility
- [Requests](https://pypi.org/project/requests/) - HTTP requests to the backend
- [Threading](https://docs.python.org/3/library/threading.html) - Concurrent video processing

---

## 🧪 How It Works

1. Start the camera via `/start-camera` endpoint.
2. YOLO model processes frames to detect faces and classify them as **real** or **fake**.
3. If the face is **real**, face recognition checks for known individuals.
4. The system saves and uploads frames if:
   - A **fake face** is detected (once per interval)
   - A **new known or unknown real face** is detected
5. You can upload new face images using `/upload-face`.

---

## 📁 Folder Structure


