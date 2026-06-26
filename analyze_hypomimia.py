import os
import cv2
import glob
import urllib.request
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Auto-download the required Google model file
model_path = 'face_landmarker.task'
if not os.path.exists(model_path):
    print("Downloading MediaPipe Face Landmarker model... (approx 9MB)")
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, model_path)
    print("Download complete!\n")

# 2. Setup the modern Tasks API
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

# Key MediaPipe landmarks for the inner mouth
TOP_LIP = 13
BOTTOM_LIP = 14
LEFT_CORNER = 78
RIGHT_CORNER = 308

def calculate_mar(landmarks):
    """Calculates the Mouth Aspect Ratio (MAR)."""
    top = np.array([landmarks[TOP_LIP].x, landmarks[TOP_LIP].y])
    bottom = np.array([landmarks[BOTTOM_LIP].x, landmarks[BOTTOM_LIP].y])
    left = np.array([landmarks[LEFT_CORNER].x, landmarks[LEFT_CORNER].y])
    right = np.array([landmarks[RIGHT_CORNER].x, landmarks[RIGHT_CORNER].y])
    
    vertical_dist = np.linalg.norm(top - bottom)
    horizontal_dist = np.linalg.norm(left - right)
    
    if horizontal_dist == 0:
        return 0.0
    return vertical_dist / horizontal_dist

def extract_visual_features(video_path):
    cap = cv2.VideoCapture(video_path)
    mar_list = []
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        # Convert BGR to RGB, then to MediaPipe Image format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect landmarks
        results = detector.detect(mp_image)
        
        if results.face_landmarks:
            landmarks = results.face_landmarks[0]
            mar = calculate_mar(landmarks)
            mar_list.append(mar)
            
    cap.release()
    
    if not mar_list:
        return 0.0, 0.0
        
    mean_mar = np.mean(mar_list)
    mar_variance = np.var(mar_list) * 1000  # Scaled by 1000 for readability
    
    return mean_mar, mar_variance

# --- Main Execution ---
print(f"{'Video ID':<15} | {'Mean MAR':<10} | {'MAR Var':<10} | {'Visual Status'}")
print("-" * 65)

videos = sorted(glob.glob("video*.mp4"))

for vid in videos:
    try:
        mean_mar, mar_var = extract_visual_features(vid)
        
        status = "Active Expression"
        if mar_var < 1.5:  
            status = "⚠️ High Masking (Rigid)"
        elif mar_var < 3.5:
            status = "⚠️ Moderate Masking"
            
        print(f"{vid:<15} | {mean_mar:.4f}   | {mar_var:.4f}     | {status}")
        
    except Exception as e:
        print(f"{vid:<15} | Error: {e}")