import os
import glob
import parselmouth
from parselmouth.praat import call
import subprocess

def extract_audio_features(video_path):
    # 1. Convert MP4 to WAV using ffmpeg (safer than reading MP4 directly)
    wav_path = video_path.replace(".mp4", ".wav")
    
    # -vn: no video, -ac 1: mono, -ar 16000: 16kHz sampling
    cmd = [
        "ffmpeg", "-y", "-i", video_path, 
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", 
        wav_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # 2. Load into Praat/Parselmouth
    sound = parselmouth.Sound(wav_path)
    
    # 3. Analyze Pitch
    pitch = sound.to_pitch()
    mean_pitch = call(pitch, "Get mean", 0, 0, "Hertz")
    
    # 4. Analyze Dysarthria Markers (Jitter & Shimmer)
    point_process = call(sound, "To PointProcess (periodic, cc)", 75, 500)
    
    # Jitter (local): Measure of pitch instability
    jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3) * 100
    
    # Shimmer (local): Measure of loudness instability
    shimmer = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6) * 100
    
    # HNR: Harmonics-to-Noise Ratio (Breathiness)
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    
    # Cleanup temp file
    if os.path.exists(wav_path):
        os.remove(wav_path)
        
    return mean_pitch, jitter, shimmer, hnr

# --- Main Execution ---
print(f"{'Video ID':<15} | {'Pitch':<8} | {'Jitter(%)':<10} | {'Shimmer(%)':<10} | {'HNR(dB)':<8} | {'Status'}")
print("-" * 85)

# Find your downloaded videos
videos = sorted(glob.glob("video*.mp4"))

for vid in videos:
    try:
        f0, jit, shim, hnr = extract_audio_features(vid)
        
        # Clinical Thresholds (Rough approximation for adult voices)
        # Jitter > 1.04% is often considered pathological
        # Shimmer > 3.81% is often considered pathological
        status = "Healthy Range"
        if jit > 1.04 or shim > 3.81:
            status = "⚠️ Potential Dysarthria"
            
        print(f"{vid:<15} | {f0:.1f}Hz   | {jit:.3f}      | {shim:.3f}       | {hnr:.1f}     | {status}")
        
    except Exception as e:
        print(f"{vid:<15} | Error: {e}")