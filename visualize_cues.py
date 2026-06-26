import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- 1. Load the Model ---
model_path = 'face_landmarker.task'
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

# --- Landmark Indices ---
# Mouth
TOP_LIP, BOTTOM_LIP = 13, 14
L_MOUTH, R_MOUTH = 78, 308

# Left Eye (Horizontal: 33 to 133, Vertical: 159 to 145)
L_EYE_TOP, L_EYE_BOT = 159, 145
L_EYE_L, L_EYE_R = 33, 133

# Right Eye (Horizontal: 362 to 263, Vertical: 386 to 374)
R_EYE_TOP, R_EYE_BOT = 386, 374
R_EYE_L, R_EYE_R = 362, 263

# Nose Tip for Tremor/Rigidity Tracking
NOSE_TIP = 4

def get_pt(lm, idx, w, h):
    """Converts normalized landmark to pixel coordinates."""
    return (int(lm[idx].x * w), int(lm[idx].y * h))

def get_dist(pt1, pt2):
    return np.linalg.norm(np.array(pt1) - np.array(pt2))

def create_dashboard(input_video, output_video):
    print(f"Opening {input_video} for advanced visualization...")
    cap = cv2.VideoCapture(input_video)
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # Store the last 30 frames of nose coordinates for the trajectory trail
    nose_trail = deque(maxlen=30)
    frame_count = 0
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        frame_count += 1
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = detector.detect(mp_image)
        
        if results.face_landmarks:
            lm = results.face_landmarks[0]
            
            # --- EXTRACT POINTS ---
            # Mouth
            pt_top_lip = get_pt(lm, TOP_LIP, width, height)
            pt_bot_lip = get_pt(lm, BOTTOM_LIP, width, height)
            pt_l_mouth = get_pt(lm, L_MOUTH, width, height)
            pt_r_mouth = get_pt(lm, R_MOUTH, width, height)
            
            # Eyes
            pt_l_eye_t = get_pt(lm, L_EYE_TOP, width, height)
            pt_l_eye_b = get_pt(lm, L_EYE_BOT, width, height)
            pt_l_eye_l = get_pt(lm, L_EYE_L, width, height)
            pt_l_eye_r = get_pt(lm, L_EYE_R, width, height)
            
            # Nose
            pt_nose = get_pt(lm, NOSE_TIP, width, height)
            nose_trail.append(pt_nose)
            
            # --- CALCULATIONS ---
            # MAR (Mouth Aspect Ratio)
            mar = get_dist(pt_top_lip, pt_bot_lip) / get_dist(pt_l_mouth, pt_r_mouth) if get_dist(pt_l_mouth, pt_r_mouth) > 0 else 0
            
            # EAR (Eye Aspect Ratio - average of both eyes)
            l_ear = get_dist(pt_l_eye_t, pt_l_eye_b) / get_dist(pt_l_eye_l, pt_l_eye_r) if get_dist(pt_l_eye_l, pt_l_eye_r) > 0 else 0
            ear = l_ear # Using left eye for simplicity, highly correlated
            
            # --- DRAWING ON FRAME ---
            # 1. Draw Mouth (Neon Green)
            cv2.line(frame, pt_top_lip, pt_bot_lip, (0, 255, 0), 2)
            cv2.line(frame, pt_l_mouth, pt_r_mouth, (0, 255, 0), 2)
            
            # 2. Draw Eye (Cyan)
            cv2.line(frame, pt_l_eye_t, pt_l_eye_b, (255, 255, 0), 2)
            cv2.line(frame, pt_l_eye_l, pt_l_eye_r, (255, 255, 0), 2)
            
            # 3. Draw Nose Trail (Red to Yellow fade)
            for i in range(1, len(nose_trail)):
                thickness = int(np.sqrt(64 / float(30 - i)) * 2)
                cv2.line(frame, nose_trail[i - 1], nose_trail[i], (0, 0, 255), thickness)
            cv2.circle(frame, pt_nose, 5, (0, 0, 255), -1) # Current nose pos
            
            # --- HUD (Heads Up Display) ---
            # Text Setup
            cv2.putText(frame, "DIAGNOSTIC VISUALIZATION (HYPOMIMIA)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # MAR Overlay (Mouth - Green)
            cv2.putText(frame, f"MAR (Speech/Bradykinesia): {mar:.3f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            mar_bar = int(mar * 300)
            cv2.rectangle(frame, (20, 95), (20 + mar_bar, 110), (0, 255, 0), -1)
            cv2.rectangle(frame, (20, 95), (170, 110), (255, 255, 255), 1)
            
            # EAR Overlay (Eyes - Cyan)
            cv2.putText(frame, f"EAR (Blink/Masking): {ear:.3f}", (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            ear_bar = int(ear * 300)
            cv2.rectangle(frame, (20, 160), (20 + ear_bar, 175), (255, 255, 0), -1)
            cv2.rectangle(frame, (20, 160), (170, 175), (255, 255, 255), 1)
            
            # Blink Detector Flag
            if ear < 0.18: # Typical blink threshold
                cv2.putText(frame, "BLINK DETECTED", (200, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
        out.write(frame)
        
    cap.release()
    out.release()
    print(f"✅ Processed {frame_count} frames.")
    print(f"Dashboard saved to: {output_video}")

# Run it
create_dashboard('aditya.mp4', 'video143cues.mp4')