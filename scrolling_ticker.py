import os
import subprocess
import shlex

# Inputs
INPUT_VIDEO = "output/left_text_with_tts.mp4"
OUTPUT_VIDEO = "output/final_with_ticker_1.mp4"
FONT_PATH = "assets/hindi-bold.ttf"
TICKER_TEXT = "ब्रेकिंग न्यूज़: उत्तर प्रदेश में मूर्तियों की ऐतिहासिक खोज से हड़कंप। प्रशासन ने जांच शुरू की।"

# Replace problematic characters
safe_text = TICKER_TEXT.replace(":", "\\:").replace("'", "’")

# Proper escaping using shlex.quote for the whole drawtext
drawtext_filter = (
    f"drawtext=fontfile={shlex.quote(FONT_PATH)}:"
    f"text={shlex.quote(safe_text)}:"
    f"fontcolor=white:fontsize=34:"
    f"x=w-mod(t*150\\,w+tw):"
    f"y=h-45:box=1:boxcolor=red@0.85:boxborderw=10"
)

cmd = [
    "ffmpeg", "-y",
    "-i", INPUT_VIDEO,
    "-vf", drawtext_filter,
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    "-c:a", "copy",
    OUTPUT_VIDEO
]

print("▶️ Adding scrolling ticker...")
try:
    subprocess.run(cmd, check=True)
    print(f"✅ Final ticker video saved: {OUTPUT_VIDEO}")
except subprocess.CalledProcessError as e:
    print(f"❌ FFmpeg failed: {e}")
