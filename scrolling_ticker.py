import subprocess
import shlex

def add_scrolling_ticker(input_video_path, output_video_path, font_path, ticker_text):
    safe_text = ticker_text.replace(":", "\\:").replace("'", "’")

    drawtext_filter = (
        f"drawtext=fontfile={shlex.quote(font_path)}:"
        f"text={shlex.quote(safe_text)}:"
        f"fontcolor=white:fontsize=48:"
        f"x=w-mod(t*150\\,w+tw):"
        f"y=h-45:box=1:boxcolor=red@0.85:boxborderw=10"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", input_video_path,
        "-vf", drawtext_filter,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        output_video_path
    ]

    print("▶️ Adding scrolling ticker...")
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Final ticker video saved: {output_video_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg failed: {e}")
