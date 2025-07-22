import os
import subprocess

# Ensure output directory
os.makedirs("output", exist_ok=True)

def normalize_clip(input_path, output_path):
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vf", "scale=1920:1080,fps=30",
        "-c:v", "libx264",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",            # ✅ Ensure AAC audio
        "-b:a", "192k",           # ✅ Reasonable bitrate
        "-ar", "44100",           # ✅ Standard sample rate
        "-ac", "2",               # ✅ Stereo
        "-movflags", "+faststart",
        output_path
    ]
    subprocess.run(cmd, check=True)

print("▶️ Normalizing clips with audio...")

normalize_clip("output/intro.mp4", "output/n_intro.mp4")
normalize_clip("output/left_text_with_tts.mp4", "output/n_left_text_with_tts.mp4")
normalize_clip("output/final_with_ticker_1.mp4", "output/n_final_with_ticker_1.mp4")

print("▶️ Creating concat list...")
with open("output/concat_list.txt", "w") as f:
    f.write("file 'n_intro.mp4'\n")
    f.write("file 'n_left_text_with_tts.mp4'\n")
    f.write("file 'n_final_with_ticker_1.mp4'\n")

print("▶️ Merging all with re-encoding...")
cmd = [
    "ffmpeg", "-y",
    "-f", "concat", "-safe", "0",
    "-i", "output/concat_list.txt",
    "-c:v", "libx264",
    "-c:a", "aac",
    "-b:a", "192k",
    "-pix_fmt", "yuv420p",
    "output/full_bulletin.mp4"
]
subprocess.run(cmd, check=True)

print("✅ Done: output/full_bulletin.mp4")
