import os
import subprocess

# Paths
BG_VIDEO = "output/bg_framed.mp4"
RIGHT_MEDIA = "assets/clips/img1.jpg"  # or .jpg/.png
OUTPUT = "output/right_section.mp4"

# Ensure output folder
os.makedirs("output", exist_ok=True)

# Detect whether it's image or video
is_image = RIGHT_MEDIA.lower().endswith((".jpg", ".png"))

# Build FFmpeg command
if is_image:
    cmd = [
        "ffmpeg",
        "-y",
        "-i", BG_VIDEO,
        "-loop", "1",
        "-t", "10",  # adjust duration as needed
        "-i", RIGHT_MEDIA,
        "-filter_complex",
        "[1:v]scale=580:400[right]; [0:v][right]overlay=670:100",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-shortest",
        OUTPUT
    ]
else:
    cmd = [
        "ffmpeg",
        "-y",
        "-i", BG_VIDEO,
        "-i", RIGHT_MEDIA,
        "-filter_complex",
        "[1:v]scale=580:400[right]; [0:v][right]overlay=670:100",
        "-map", "0:a?",  # use audio from bg if exists
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-shortest",
        OUTPUT
    ]

# Run command
print("▶️ Adding right-side media...")
subprocess.run(cmd, check=True)
print(f"✅ Right-side section video saved: {OUTPUT}")
