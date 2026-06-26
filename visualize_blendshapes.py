import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- 1. Load the Model ---
model_path = 'face_landmarker.task'
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=False,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

def get_pt(lm, idx, w, h):
    """Helper to convert normalized landmarks to pixel coordinates."""
    return (int(lm[idx].x * w), int(lm[idx].y * h))

def create_muscle_tracking_dashboard(input_video, output_video):
    print(f"Tracking raw muscle activation on {input_video}...")
    cap = cv2.VideoCapture(input_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = detector.detect(mp_image)
        
        # We need BOTH Blendshapes (for the bars) and Landmarks (for the green lines)
        if results.face_blendshapes and results.face_landmarks:
            blendshapes = results.face_blendshapes[0]
            landmarks = results.face_landmarks[0]
            scores = {category.category_name: category.score for category in blendshapes}
            
            # --- EXTRACT RAW SCORES ---
            zygomatic_score = (scores.get('mouthSmileLeft', 0) + scores.get('mouthSmileRight', 0)) / 2.0
            orbicularis_score = (scores.get('eyeSquintLeft', 0) + scores.get('eyeSquintRight', 0)) / 2.0
            brow_score = (scores.get('browInnerUp', 0) + scores.get('browDownLeft', 0)) / 2.0

            # --- DRAW ANATOMICAL MUSCLE LINES (NEON GREEN) ---
            # 1. Zygomaticus Major Vectors (Connects mouth corners to cheekbones)
            # Left side
            cv2.line(frame, get_pt(landmarks, 61, width, height), get_pt(landmarks, 205, width, height), (0, 255, 0), 2)
            # Right side
            cv2.line(frame, get_pt(landmarks, 291, width, height), get_pt(landmarks, 425, width, height), (0, 255, 0), 2)
            
            # 2. Corrugator / Frontalis (Trace the eyebrows)
            left_brow = [46, 53, 52, 65, 55]
            right_brow = [276, 283, 282, 295, 285]
            for i in range(len(left_brow) - 1):
                cv2.line(frame, get_pt(landmarks, left_brow[i], width, height), get_pt(landmarks, left_brow[i+1], width, height), (0, 255, 0), 2)
                cv2.line(frame, get_pt(landmarks, right_brow[i], width, height), get_pt(landmarks, right_brow[i+1], width, height), (0, 255, 0), 2)

            # 3. Orbicularis Oculi (Outline the eyes to show squinting/widening)
            left_eye = [33, 160, 158, 133, 153, 144, 33]
            right_eye = [362, 385, 387, 263, 373, 380, 362]
            for i in range(len(left_eye) - 1):
                cv2.line(frame, get_pt(landmarks, left_eye[i], width, height), get_pt(landmarks, left_eye[i+1], width, height), (0, 255, 0), 1)
                cv2.line(frame, get_pt(landmarks, right_eye[i], width, height), get_pt(landmarks, right_eye[i+1], width, height), (0, 255, 0), 1)

            # --- HUD DRAWING (NO THRESHOLDS) ---
            cv2.rectangle(frame, (10, 20), (450, 210), (0, 0, 0), -1)
            cv2.putText(frame, "RAW MUSCLE ACTIVATION TRACKER", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            def draw_raw_bar(y_pos, label, score, color):
                cv2.putText(frame, f"{label}: {score:.3f}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                bar_len = int(score * 350)
                cv2.rectangle(frame, (20, y_pos + 10), (20 + bar_len, y_pos + 25), color, -1)
                cv2.rectangle(frame, (20, y_pos + 10), (370, y_pos + 25), (255, 255, 255), 1)

            draw_raw_bar(90,  "Zygomaticus (Smile/Cheek)", zygomatic_score, (0, 255, 0))
            draw_raw_bar(140, "Orbicularis (Eye Squint)", orbicularis_score, (255, 255, 0))
            draw_raw_bar(190, "Corrugator (Brow Furrow)", brow_score, (0, 165, 255))

        out.write(frame)
        
    cap.release()
    out.release()
    print(f"✅ Raw Muscle Analysis saved to: {output_video}")

# Run it on your video
create_muscle_tracking_dashboard('video148.mp4', 'video_muscle_tracking.mp4')