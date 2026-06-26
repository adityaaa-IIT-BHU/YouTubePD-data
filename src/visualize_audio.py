import cv2
import parselmouth
from parselmouth.praat import call
import numpy as np
import subprocess
import os
import math

def get_rolling_metrics(sound, t, window=1.0):
    """Calculates Jitter & Shimmer for a 1-second window around time t."""
    start = max(0, t - window/2)
    end = min(sound.get_total_duration(), t + window/2)
    
    # If the window is too small, return 0
    if end - start < 0.1: 
        return 0, 0
        
    part = sound.extract_part(from_time=start, to_time=end, preserve_times=True)
    point_process = call(part, "To PointProcess (periodic, cc)", 75, 500)
    
    try:
        # Praat throws errors if there are no voiced frames in the window (e.g., silence)
        jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3) * 100
        shimmer = call([part, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6) * 100
        if math.isnan(jitter): jitter = 0
        if math.isnan(shimmer): shimmer = 0
    except:
        jitter, shimmer = 0, 0
        
    return jitter, shimmer

def create_realtime_audio_dashboard(video_path, output_path):
    print(f"Extracting audio from {video_path}...")
    temp_wav = "temp_audio.wav"
    temp_vid = "temp_video.mp4"
    
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path, 
        "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1", temp_wav
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    print("Loading audio into Praat...")
    sound = parselmouth.Sound(temp_wav)
    pitch = sound.to_pitch()
    intensity = sound.to_intensity()

    print("Generating visual overlays (This takes a bit longer due to rolling windows)...")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_vid, fourcc, fps, (width, height))
    
    frame_num = 0
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        t = frame_num / fps
        
        # 1. Instantaneous Metrics
        try:
            p_val = pitch.get_value_at_time(t)
            i_val = intensity.get_value(t)
            if math.isnan(p_val): p_val = 0
            if math.isnan(i_val): i_val = 0
        except:
            p_val, i_val = 0, 0
            
        # 2. Rolling Window Metrics (Real-time Jitter/Shimmer)
        j_val, s_val = get_rolling_metrics(sound, t, window=1.0)
            
        # --- DRAW DASHBOARD ---
        # Larger background panel for 4 metrics
        cv2.rectangle(frame, (20, 20), (550, 260), (0, 0, 0), -1)
        cv2.putText(frame, "REAL-TIME DYSARTHRIA TRACKER", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Helper function for drawing bars
        def draw_bar(y_pos, label, val, max_val, color, threshold=None):
            cv2.putText(frame, f"{label}: {val:.2f}", (30, y_pos+15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            bar_len = min(int((val / max_val) * 250), 250)
            cv2.rectangle(frame, (260, y_pos), (260 + bar_len, y_pos + 20), color, -1)
            cv2.rectangle(frame, (260, y_pos), (260 + 250, y_pos + 20), (255, 255, 255), 1)
            
            # Draw clinical threshold line if provided
            if threshold:
                thresh_x = 260 + int((threshold / max_val) * 250)
                cv2.line(frame, (thresh_x, y_pos - 5), (thresh_x, y_pos + 25), (255, 0, 0), 2) # Blue threshold line

        # Draw the 4 dynamic bars
        draw_bar(80,  "Pitch (Hz)", p_val, 300.0, (0, 255, 255))               # Yellow
        draw_bar(120, "Volume (dB)", i_val, 100.0, (0, 255, 0))                # Green
        
        # Jitter (>1.04% is pathological) - Scaled to max 5.0% for visibility
        draw_bar(160, "Jitter (%)", j_val, 5.0, (0, 0, 255), threshold=1.04)   # Red
        
        # Shimmer (>3.81% is pathological) - Scaled to max 15.0% for visibility
        draw_bar(200, "Shimmer (%)", s_val, 15.0, (0, 165, 255), threshold=3.81) # Orange
        
        # Legend
        cv2.putText(frame, "Blue Line = Healthy Clinical Threshold", (30, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1)

        out.write(frame)
        frame_num += 1

    cap.release()
    out.release()
    
    print("Stitching audio back into the video dashboard...")
    subprocess.run([
        "ffmpeg", "-y", "-i", temp_vid, "-i", temp_wav,
        "-c:v", "copy", "-c:a", "aac", output_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    if os.path.exists(temp_wav): os.remove(temp_wav)
    if os.path.exists(temp_vid): os.remove(temp_vid)
    print(f"✅ Real-time audio dashboard complete: {output_path}")

# Run it
create_realtime_audio_dashboard('aditya.mp4', '143audio.mp4')