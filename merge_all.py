import os
import subprocess

# List of clips to merge
clips = [
    "output/intro.mp4",
    "output/final_with_ticker_1.mp4",
    
    
]

concat_list_path = "output/concat_list.txt"
with open(concat_list_path, "w") as f:
    for clip in clips:
        f.write(f"file '{os.path.abspath(clip)}'\n")

# Output file
output_path = "output/full_bulletin.mp4"

# Run FFmpeg concat
cmd = [
    "ffmpeg", "-y",
    "-f", "concat",
    "-safe", "0",
    "-i", concat_list_path,
    "-c", "copy",
    output_path
]

print("▶️ Merging all clips...")
subprocess.run(cmd, check=True)
print(f"✅ Full bulletin saved: {output_path}")
