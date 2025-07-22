import os
import subprocess

BACKGROUND = "assets/bg/earth_bg.mp4"
FRAME = "assets/frame.png"
OUTPUT = "output/bg_framed.mp4"

os.makedirs("output", exist_ok=True)

cmd = [
    "ffmpeg",
    "-y",
    "-i", BACKGROUND,
    "-i", FRAME,
    "-filter_complex",
    "[0:v]scale=1280:720[bg]; [1:v]scale=1280:720[frame]; [bg][frame]overlay=0:0[out]",
    "-map", "[out]",
    "-map", "0:a?",  # include audio only if background has it
    "-c:v", "libx264",
    "-c:a", "aac",
    "-pix_fmt", "yuv420p",
    OUTPUT
]

print("▶️ Generating base with background + frame...")
subprocess.run(cmd, check=True)
print(f"✅ Base saved: {OUTPUT}")
