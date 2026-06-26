import os
import pandas as pd
import subprocess
import time

def get_sec(time_str):
    """Robustly handles 'HH:MM:SS' and 'MM:SS' and returns total seconds."""
    time_str = str(time_str).strip()
    if " " in time_str:
        time_str = time_str.split(" ")[-1]
        
    parts = time_str.split(':')
    try:
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + int(float(s))
        elif len(parts) == 2:
            m, s = parts
            return int(m) * 60 + int(float(s))
        elif len(parts) == 1:
            return int(float(parts[0]))
    except ValueError:
        return 0
    return 0

def smart_timestamp_fix(seconds, duration, raw_str):
    """Corrects Pandas HH:MM:SS reading errors if it exceeds video duration."""
    if seconds < duration:
        return seconds
        
    parts = str(raw_str).strip().split(':')
    if len(parts) == 3:
        shifted_seconds = int(parts[0]) * 60 + int(parts[1])
        if shifted_seconds < duration:
            print(f"   ↳ ⚠️ Auto-Correction: Interpreting '{raw_str}' as MM:SS ({shifted_seconds}s).")
            return shifted_seconds
    return seconds

print("Reading Excel file...")
df = pd.read_excel('data_sheets/data_sheet.xlsx')
total_rows = len(df)
print(f"Found {total_rows} total videos in the dataset.")

# Loop through the ENTIRE dataset instead of just range(10)
for i in range(total_rows): 
    try:
        name = "video" + str(i + 134)
        filename = f"{name}.mp4"
        
        # --- BULK LOGIC: Skip if already downloaded ---
        if os.path.exists(filename) and os.path.getsize(filename) > 1000:
            print(f"⏭️ {filename} already exists. Skipping...")
            continue
            
        raw_start = str(df.start[i])
        raw_end = str(df.end[i])
        link = str(df.link[i])
        
        # Skip empty rows
        if link == 'nan' or pd.isna(link): 
            continue
        
        print(f"\nProcessing {name} (Link: {link})")

        # Step 1: Check Video Duration
        cmd_dur = ["yt-dlp", "--print", "duration", link]
        proc_dur = subprocess.run(cmd_dur, capture_output=True, text=True)
        
        try:
            total_duration = float(proc_dur.stdout.strip())
        except ValueError:
            print(f"⚠️ Could not get duration (Video might be deleted/private). Skipping.")
            continue

        # Step 2: Calculate and Fix Timestamps
        start_sec = get_sec(raw_start)
        start_sec = smart_timestamp_fix(start_sec, total_duration, raw_start)
        
        end_sec = get_sec(raw_end)
        end_sec = smart_timestamp_fix(end_sec, total_duration, raw_end)

        if start_sec >= total_duration:
            print(f"❌ ERROR: Start time ({start_sec}s) > Duration ({total_duration}s). Skipping.")
            continue
            
        if end_sec > total_duration:
            print(f"   ↳ Clamping end time to video duration ({total_duration}s)")
            end_sec = total_duration

        print(f"   ↳ Segment: {start_sec}s to {end_sec}s")

        # Step 3: Download
        section_arg = f"*{start_sec}-{end_sec}"
        cmd = [
            "yt-dlp",
            "--force-keyframes-at-cuts", 
            "--download-sections", section_arg,
            "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "-o", filename,
            "--force-overwrites",
            link
        ]
        
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        if process.returncode == 0:
            if os.path.exists(filename) and os.path.getsize(filename) > 1000:
                print(f"✅ Success: {filename} created")
            else:
                print(f"⚠️ Failed: File is empty or missing.")
        else:
            print(f"❌ yt-dlp Error on {name}")

        # --- BULK LOGIC: Anti-Ban Delay ---
        time.sleep(3)

    except Exception as e:
        print(f"❌ Script Error on {name}: {e}")
        pass

print("\n--- 🏁 BULK DOWNLOAD SCRIPT FINISHED ---")