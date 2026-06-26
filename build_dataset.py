import os
import glob
import cv2
import numpy as np
import pandas as pd
import subprocess
import parselmouth
from parselmouth.praat import call
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- 1. Setup MediaPipe ---
model_path = 'face_landmarker.task'
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

# --- 2. Feature Extraction Functions ---
def extract_audio_features(video_path):
    wav_path = video_path.replace(".mp4", ".wav")
    cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", wav_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    try:
        sound = parselmouth.Sound(wav_path)
        pitch = sound.to_pitch()
        f0 = call(pitch, "Get mean", 0, 0, "Hertz")
        
        point_process = call(sound, "To PointProcess (periodic, cc)", 75, 500)
        jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3) * 100
        shimmer = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6) * 100
        
        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = call(harmonicity, "Get mean", 0, 0)
    except:
        f0, jitter, shimmer, hnr = np.nan, np.nan, np.nan, np.nan
        
    if os.path.exists(wav_path): os.remove(wav_path)
    return f0, jitter, shimmer, hnr

def extract_visual_features(video_path):
    cap = cv2.VideoCapture(video_path)
    mar_list = []
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = detector.detect(mp_image)
        
        if results.face_landmarks:
            lm = results.face_landmarks[0]
            top, bottom = np.array([lm[13].x, lm[13].y]), np.array([lm[14].x, lm[14].y])
            left, right = np.array([lm[78].x, lm[78].y]), np.array([lm[308].x, lm[308].y])
            
            v_dist = np.linalg.norm(top - bottom)
            h_dist = np.linalg.norm(left - right)
            if h_dist > 0:
                mar_list.append(v_dist / h_dist)
                
    cap.release()
    if not mar_list: return np.nan, np.nan
    return np.mean(mar_list), np.var(mar_list) * 1000

# --- 3. Main Loop ---
print("Extracting Multimodal Features. This will take a moment...\n")
data = []
videos = sorted(glob.glob("video*.mp4"))

for vid in videos:
    print(f"Processing {vid}...")
    f0, jit, shim, hnr = extract_audio_features(vid)
    mean_mar, var_mar = extract_visual_features(vid)
    
    data.append({
        "Video_ID": vid,
        "F0_Hz": f0,
        "Jitter_Pct": jit,
        "Shimmer_Pct": shim,
        "HNR_dB": hnr,
        "MAR_Mean": mean_mar,
        "MAR_Variance": var_mar
    })

# Save to CSV
df = pd.DataFrame(data)
df.to_csv("multimodal_features.csv", index=False)
print("\n✅ Success! Data saved to 'multimodal_features.csv'")