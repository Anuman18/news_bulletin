import os
import subprocess
import textwrap
import requests
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from moviepy.editor import VideoFileClip, ImageClip, concatenate_videoclips
from mutagen.mp3 import MP3
from google.cloud import texttospeech

app = FastAPI()

# Setup paths
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key.json"
TMP_DIR = "temp"
OUTPUT_DIR = "output"
FONT_PATH = "assets/hindi-bold.ttf"
os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Request models
class ContentItem(BaseModel):
    segment_type: str  # "video_with_text" or "video_without_text"
    media_url: str
    text: Optional[str] = None

class BulletinRequest(BaseModel):
    language_code: str
    language_name: str
    ticker: str
    frame_url: str
    background_url: str
    content: List[ContentItem]

# === Helper functions ===
def download_file(url, filename):
    path = os.path.join(TMP_DIR, filename)
    r = requests.get(url)
    if r.status_code == 200:
        with open(path, "wb") as f:
            f.write(r.content)
        return path
    raise Exception(f"Failed to download: {url}")

def generate_tts(text, voice_name, lang_code, output_path):
    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(language_code=lang_code, name=voice_name)
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
    with open(output_path, "wb") as out:
        out.write(response.audio_content)

def get_audio_duration(audio_path):
    return MP3(audio_path).info.length

def get_video_duration(video_path):
    """Get duration of video file"""
    cmd = [
        "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())

def extend_media(media_path, duration, output_path):
    ext = os.path.splitext(media_path)[1].lower()
    if ext in ['.jpg', '.jpeg', '.png']:
        clip = ImageClip(media_path).set_duration(duration).resize(height=650).set_position(("right", "center"))
    else:
        clip = VideoFileClip(media_path)
        loops = int(duration // clip.duration) + 1
        final = concatenate_videoclips([clip] * loops).subclip(0, duration)
        clip = final.resize(height=650).set_position(("right", "center"))
    clip.write_videofile(output_path, fps=25, codec="libx264")

def create_text_background(bg_path, text, audio_path, output_path):
    """Create background video with text overlay and TTS audio"""
    font_size = 52
    font_path = os.path.abspath(FONT_PATH)
    subtitle_file = os.path.join(TMP_DIR, "tts_text.txt")

    wrapped_text = textwrap.fill(text, width=24)
    with open(subtitle_file, "w", encoding="utf-8") as f:
        f.write(wrapped_text)

    # Get audio duration to set video duration
    audio_duration = get_audio_duration(audio_path)
    
    assert os.path.exists(bg_path), "Missing background"
    assert os.path.exists(audio_path), "Missing TTS audio"
    assert os.path.exists(font_path), "Missing font"
    assert os.path.exists(subtitle_file), "Subtitle file missing"

    # First create a looped background video with exact duration
    temp_bg_looped = os.path.join(TMP_DIR, "bg_looped_temp.mp4")
    loop_cmd = [
        "ffmpeg", "-y",
        "-stream_loop", "-1",  # Loop indefinitely
        "-i", bg_path,
        "-t", str(audio_duration),  # Set exact duration
        "-c:v", "libx264",
        temp_bg_looped
    ]
    subprocess.run(loop_cmd, check=True)

    # Then add text overlay and audio
    filter_str = (
        f"[0:v]drawbox=x=40:y=100:w=600:h=400:color=black@0.7:t=fill,"
        f"drawtext=fontfile={font_path}:textfile={subtitle_file}:"
        f"fontcolor=white:fontsize={font_size}:x=60:y=140:"
        f"shadowcolor=black:shadowx=2:shadowy=2[v]"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", temp_bg_looped,
        "-i", audio_path,
        "-filter_complex", filter_str,
        "-map", "[v]",
        "-map", "1:a",  # Map audio from second input
        "-c:v", "libx264",
        "-c:a", "aac",
        output_path
    ]

    print("🛠 Running FFmpeg command:\n", " ".join(cmd))
    subprocess.run(cmd, check=True)

def overlay_right_and_frame(bg_text_video, right_video, frame_img, output_path):
    """Overlay right video and frame, preserving audio duration"""
    # Get the duration of the background video (which has TTS audio)
    bg_duration = get_video_duration(bg_text_video)
    
    # First create a looped right video with exact duration
    temp_right_looped = os.path.join(TMP_DIR, "right_looped_temp.mp4")
    loop_cmd = [
        "ffmpeg", "-y",
        "-stream_loop", "-1",  # Loop indefinitely
        "-i", right_video,
        "-t", str(bg_duration),  # Set exact duration
        "-c:v", "libx264",
        temp_right_looped
    ]
    subprocess.run(loop_cmd, check=True)
    
    cmd = [
        "ffmpeg", "-y",
        "-i", bg_text_video,
        "-i", temp_right_looped,
        "-i", frame_img,
        "-filter_complex",
        "[0:v][1:v]overlay=W-w-50:(H-h)/2[tmp];"
        "[tmp][2:v]overlay=0:0[out]",
        "-map", "[out]", 
        "-map", "0:a",  # Keep original audio from bg_text_video
        "-c:v", "libx264", 
        "-c:a", "aac", 
        output_path
    ]
    subprocess.run(cmd, check=True)

def process_video_without_text(media_path, output_path):
    """Process video that doesn't need TTS - just copy with proper audio"""
    cmd = [
        "ffmpeg", "-y",
        "-i", media_path,
        "-c:v", "libx264",
        "-c:a", "aac",
        output_path
    ]
    subprocess.run(cmd, check=True)

# Updated concat function to handle audio properly
def concat_videos(video_paths, output_path):
    """Concatenate videos ensuring all audio tracks are preserved"""
    # Create filter complex for concatenation
    inputs = []
    filter_parts = []
    
    for i, video in enumerate(video_paths):
        inputs.extend(["-i", video])
        filter_parts.append(f"[{i}:v][{i}:a]")
    
    filter_complex = "".join(filter_parts) + f"concat=n={len(video_paths)}:v=1:a=1[outv][outa]"
    
    cmd = [
        "ffmpeg", "-y"
    ] + inputs + [
        "-filter_complex", filter_complex,
        "-map", "[outv]",
        "-map", "[outa]",
        "-c:v", "libx264",
        "-c:a", "aac",
        output_path
    ]
    
    print("🛠 Concatenating videos with command:\n", " ".join(cmd))
    subprocess.run(cmd, check=True)

def add_ticker(input_video, ticker_text, output_path):
    repeated = (ticker_text + "   ") * 3
    ticker_file = os.path.join(TMP_DIR, "ticker.txt")
    with open(ticker_file, "w", encoding="utf-8") as f:
        f.write(repeated)

    cmd = [
        "ffmpeg", "-y", "-i", input_video,
        "-vf", f"drawbox=y=ih-60:color=red@1.0:width=iw:height=60:t=fill,"
               f"drawtext=fontfile={FONT_PATH}:textfile={ticker_file}:"
               f"fontsize=42:fontcolor=white:x=w-mod(t*200\\,w+tw):y=h-50:"
               f"shadowcolor=black:shadowx=2:shadowy=2",
        "-codec:a", "copy",  # Copy audio without re-encoding
        output_path
    ]
    subprocess.run(cmd, check=True)

# === API ===
@app.post("/generate")
def generate_bulletin(req: BulletinRequest):
    try:
        bg_path = download_file(req.background_url, "bg.mp4")
        frame_path = download_file(req.frame_url, "frame.png")
        segments = []

        for i, item in enumerate(req.content):
            media_path = download_file(item.media_url, f"media_{i}.mp4")

            if item.segment_type == "video_without_text":
                # Process video without text but ensure proper encoding
                processed_video = os.path.join(TMP_DIR, f"processed_{i}.mp4")
                process_video_without_text(media_path, processed_video)
                segments.append(processed_video)

            elif item.segment_type == "video_with_text":
                tts_path = os.path.join(TMP_DIR, f"tts_{i}.mp3")
                generate_tts(item.text, req.language_name, req.language_code, tts_path)
                duration = get_audio_duration(tts_path)

                right_clip = os.path.join(TMP_DIR, f"right_{i}.mp4")
                text_bg = os.path.join(TMP_DIR, f"textbg_{i}.mp4")
                final_clip = os.path.join(TMP_DIR, f"final_{i}.mp4")

                extend_media(media_path, duration, right_clip)
                create_text_background(bg_path, item.text, tts_path, text_bg)
                overlay_right_and_frame(text_bg, right_clip, frame_path, final_clip)

                segments.append(final_clip)

        # Concatenate all segments
        merged_video = os.path.join(OUTPUT_DIR, "merged.mp4")
        concat_videos(segments, merged_video)

        # Add ticker
        final_output = os.path.join(OUTPUT_DIR, "final_with_ticker.mp4")
        add_ticker(merged_video, req.ticker, final_output)

        return {"status": "success", "output_path": final_output}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# === Run with uvicorn ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

    
import os
import subprocess
import textwrap
import requests
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from moviepy.editor import VideoFileClip, ImageClip, concatenate_videoclips
from mutagen.mp3 import MP3
from google.cloud import texttospeech

app = FastAPI()

# Setup paths
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key.json"
TMP_DIR = "temp"
OUTPUT_DIR = "output"
FONT_PATH = "assets/hindi-bold.ttf"
os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Request models
class ContentItem(BaseModel):
    segment_type: str  # "video_with_text" or "video_without_text"
    media_url: str
    text: Optional[str] = None

class BulletinRequest(BaseModel):
    language_code: str
    language_name: str
    ticker: str
    frame_url: str
    background_url: str
    content: List[ContentItem]

# === Helper functions ===
def download_file(url, filename):
    path = os.path.join(TMP_DIR, filename)
    r = requests.get(url)
    if r.status_code == 200:
        with open(path, "wb") as f:
            f.write(r.content)
        return path
    raise Exception(f"Failed to download: {url}")

def generate_tts(text, voice_name, lang_code, output_path):
    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(language_code=lang_code, name=voice_name)
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
    with open(output_path, "wb") as out:
        out.write(response.audio_content)

def get_audio_duration(audio_path):
    return MP3(audio_path).info.length

def get_video_duration(video_path):
    """Get duration of video file"""
    cmd = [
        "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())

def extend_media(media_path, duration, output_path):
    ext = os.path.splitext(media_path)[1].lower()
    if ext in ['.jpg', '.jpeg', '.png']:
        clip = ImageClip(media_path).set_duration(duration).resize(height=650).set_position(("right", "center"))
    else:
        clip = VideoFileClip(media_path)
        loops = int(duration // clip.duration) + 1
        final = concatenate_videoclips([clip] * loops).subclip(0, duration)
        clip = final.resize(height=650).set_position(("right", "center"))
    clip.write_videofile(output_path, fps=25, codec="libx264")

def create_text_background(bg_path, text, audio_path, output_path):
    """Create background video with text overlay and TTS audio"""
    font_size = 52
    font_path = os.path.abspath(FONT_PATH)
    subtitle_file = os.path.join(TMP_DIR, "tts_text.txt")

    wrapped_text = textwrap.fill(text, width=24)
    with open(subtitle_file, "w", encoding="utf-8") as f:
        f.write(wrapped_text)

    # Get audio duration to set video duration
    audio_duration = get_audio_duration(audio_path)
    
    assert os.path.exists(bg_path), "Missing background"
    assert os.path.exists(audio_path), "Missing TTS audio"
    assert os.path.exists(font_path), "Missing font"
    assert os.path.exists(subtitle_file), "Subtitle file missing"

    # First create a looped background video with exact duration
    temp_bg_looped = os.path.join(TMP_DIR, "bg_looped_temp.mp4")
    loop_cmd = [
        "ffmpeg", "-y",
        "-stream_loop", "-1",  # Loop indefinitely
        "-i", bg_path,
        "-t", str(audio_duration),  # Set exact duration
        "-c:v", "libx264",
        temp_bg_looped
    ]
    subprocess.run(loop_cmd, check=True)

    # Then add text overlay and audio
    filter_str = (
        f"[0:v]drawbox=x=40:y=100:w=600:h=400:color=black@0.7:t=fill,"
        f"drawtext=fontfile={font_path}:textfile={subtitle_file}:"
        f"fontcolor=white:fontsize={font_size}:x=60:y=140:"
        f"shadowcolor=black:shadowx=2:shadowy=2[v]"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", temp_bg_looped,
        "-i", audio_path,
        "-filter_complex", filter_str,
        "-map", "[v]",
        "-map", "1:a",  # Map audio from second input
        "-c:v", "libx264",
        "-c:a", "aac",
        output_path
    ]

    print("🛠 Running FFmpeg command:\n", " ".join(cmd))
    subprocess.run(cmd, check=True)

def overlay_right_and_frame(bg_text_video, right_video, frame_img, output_path):
    """Overlay right video and frame, preserving audio duration"""
    # Get the duration of the background video (which has TTS audio)
    bg_duration = get_video_duration(bg_text_video)
    
    # First create a looped right video with exact duration
    temp_right_looped = os.path.join(TMP_DIR, "right_looped_temp.mp4")
    loop_cmd = [
        "ffmpeg", "-y",
        "-stream_loop", "-1",  # Loop indefinitely
        "-i", right_video,
        "-t", str(bg_duration),  # Set exact duration
        "-c:v", "libx264",
        temp_right_looped
    ]
    subprocess.run(loop_cmd, check=True)
    
    cmd = [
        "ffmpeg", "-y",
        "-i", bg_text_video,
        "-i", temp_right_looped,
        "-i", frame_img,
        "-filter_complex",
        "[0:v][1:v]overlay=W-w-50:(H-h)/2[tmp];"
        "[tmp][2:v]overlay=0:0[out]",
        "-map", "[out]", 
        "-map", "0:a",  # Keep original audio from bg_text_video
        "-c:v", "libx264", 
        "-c:a", "aac", 
        output_path
    ]
    subprocess.run(cmd, check=True)

def process_video_without_text(media_path, output_path):
    """Process video that doesn't need TTS - just copy with proper audio"""
    cmd = [
        "ffmpeg", "-y",
        "-i", media_path,
        "-c:v", "libx264",
        "-c:a", "aac",
        output_path
    ]
    subprocess.run(cmd, check=True)

# Updated concat function to handle audio properly
def concat_videos(video_paths, output_path):
    """Concatenate videos ensuring all audio tracks are preserved"""
    # Create filter complex for concatenation
    inputs = []
    filter_parts = []
    
    for i, video in enumerate(video_paths):
        inputs.extend(["-i", video])
        filter_parts.append(f"[{i}:v][{i}:a]")
    
    filter_complex = "".join(filter_parts) + f"concat=n={len(video_paths)}:v=1:a=1[outv][outa]"
    
    cmd = [
        "ffmpeg", "-y"
    ] + inputs + [
        "-filter_complex", filter_complex,
        "-map", "[outv]",
        "-map", "[outa]",
        "-c:v", "libx264",
        "-c:a", "aac",
        output_path
    ]
    
    print("🛠 Concatenating videos with command:\n", " ".join(cmd))
    subprocess.run(cmd, check=True)

def add_ticker(input_video, ticker_text, output_path):
    repeated = (ticker_text + "   ") * 3
    ticker_file = os.path.join(TMP_DIR, "ticker.txt")
    with open(ticker_file, "w", encoding="utf-8") as f:
        f.write(repeated)

    cmd = [
        "ffmpeg", "-y", "-i", input_video,
        "-vf", f"drawbox=y=ih-60:color=red@1.0:width=iw:height=60:t=fill,"
               f"drawtext=fontfile={FONT_PATH}:textfile={ticker_file}:"
               f"fontsize=42:fontcolor=white:x=w-mod(t*200\\,w+tw):y=h-50:"
               f"shadowcolor=black:shadowx=2:shadowy=2",
        "-codec:a", "copy",  # Copy audio without re-encoding
        output_path
    ]
    subprocess.run(cmd, check=True)

# === API ===
@app.post("/generate")
def generate_bulletin(req: BulletinRequest):
    try:
        bg_path = download_file(req.background_url, "bg.mp4")
        frame_path = download_file(req.frame_url, "frame.png")
        segments = []

        for i, item in enumerate(req.content):
            media_path = download_file(item.media_url, f"media_{i}.mp4")

            if item.segment_type == "video_without_text":
                # Process video without text but ensure proper encoding
                processed_video = os.path.join(TMP_DIR, f"processed_{i}.mp4")
                process_video_without_text(media_path, processed_video)
                segments.append(processed_video)

            elif item.segment_type == "video_with_text":
                tts_path = os.path.join(TMP_DIR, f"tts_{i}.mp3")
                generate_tts(item.text, req.language_name, req.language_code, tts_path)
                duration = get_audio_duration(tts_path)

                right_clip = os.path.join(TMP_DIR, f"right_{i}.mp4")
                text_bg = os.path.join(TMP_DIR, f"textbg_{i}.mp4")
                final_clip = os.path.join(TMP_DIR, f"final_{i}.mp4")

                extend_media(media_path, duration, right_clip)
                create_text_background(bg_path, item.text, tts_path, text_bg)
                overlay_right_and_frame(text_bg, right_clip, frame_path, final_clip)

                segments.append(final_clip)

        # Concatenate all segments
        merged_video = os.path.join(OUTPUT_DIR, "merged.mp4")
        concat_videos(segments, merged_video)

        # Add ticker
        final_output = os.path.join(OUTPUT_DIR, "final_with_ticker.mp4")
        add_ticker(merged_video, req.ticker, final_output)

        return {"status": "success", "output_path": final_output}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# === Run with uvicorn ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)