import os
import subprocess
import textwrap
import requests
import time
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
VIDEO_BULLETIN_DIR = "video_bulletin"
THUMBNAIL_DIR = "thumbnail"
FONT_PATH = "assets/NotoSans-Regular.ttf"
os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIDEO_BULLETIN_DIR, exist_ok=True)
os.makedirs(THUMBNAIL_DIR, exist_ok=True)

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
        clip = ImageClip(media_path).set_duration(duration).resize(width=1100, height=620).set_position("center")
    else:
        clip = VideoFileClip(media_path)
        loops = int(duration // clip.duration) + 1
        final = concatenate_videoclips([clip] * loops).subclip(0, duration)
        clip = final.resize(width=1100, height=620).set_position("center")
    clip.write_videofile(output_path, fps=25, codec="libx264")

def create_text_background(bg_path, text, audio_path, output_path):
    """Create background video with text overlay and TTS audio - MODIFIED FOR LEFT SIDE TEXT"""
    font_size = 46
    font_path = os.path.abspath(FONT_PATH)
    subtitle_file = os.path.join(TMP_DIR, "tts_text.txt")
    breaking_news_file = os.path.join(TMP_DIR, "breaking_news.txt")

    # Fixed box dimensions - aligned with right frame
    box_width = 650
    box_height = 460  
    box_y = 220  # Moved down to align with right frame
    header_y = box_y + 10  # Breaking News header position
    text_y = box_y + 70  # Text starts below the header
    
    # Write "BREAKING NEWS" text
    with open(breaking_news_file, "w", encoding="utf-8") as f:
        f.write("BREAKING NEWS")
    
    # Wrap text to fit in the box
    wrapped_text = textwrap.fill(text, width=28)
    
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

    # Create complex filter with Breaking News header and text
    filter_str = (
        # Draw main black box
        f"[0:v]drawbox=x=40:y={box_y}:w={box_width}:h={box_height}:color=black@0.7:t=fill,"
        # Draw red box for Breaking News header
        f"drawbox=x=40:y={box_y}:w={box_width}:h=50:color=red@1.0:t=fill,"
        # Draw Breaking News text in white
        f"drawtext=fontfile={font_path}:textfile={breaking_news_file}:"
        f"fontcolor=white:fontsize=36:x=(40+{box_width}/2-text_w/2):y={header_y}:"
        f"shadowcolor=black:shadowx=2:shadowy=2,"
        # Draw main text
        f"drawtext=fontfile={font_path}:textfile={subtitle_file}:"
        f"fontcolor=white:fontsize={font_size}:x=60:y={text_y}:"
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
    
    # Check if right_video is an image or video
    ext = os.path.splitext(right_video)[1].lower()
    
    if ext in ['.jpg', '.jpeg', '.png']:
        # For images, create a video from the image
        loop_cmd = [
            "ffmpeg", "-y",
            "-loop", "1",
            "-i", right_video,
            "-t", str(bg_duration),
            "-vf", "scale=1100:620",  # Fixed size for right video
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            temp_right_looped
        ]
    else:
        # For videos, loop them
        loop_cmd = [
            "ffmpeg", "-y",
            "-stream_loop", "-1",  # Loop indefinitely
            "-i", right_video,
            "-t", str(bg_duration),  # Set exact duration
            "-vf", "scale=1100:620",  # Fixed size for right video
            "-c:v", "libx264",
            temp_right_looped
        ]
    
    subprocess.run(loop_cmd, check=True)
    
    # Overlay with better positioning
    cmd = [
        "ffmpeg", "-y",
        "-i", bg_text_video,
        "-i", temp_right_looped,
        "-i", frame_img,
        "-filter_complex",
        "[0:v][1:v]overlay=760:220[tmp];"  # Fixed position for right video
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
    """Add two tickers - one at top and one at bottom"""
    repeated_bottom = (ticker_text + "   ") * 3
    ticker_file_bottom = os.path.join(TMP_DIR, "ticker_bottom.txt")
    with open(ticker_file_bottom, "w", encoding="utf-8") as f:
        f.write(repeated_bottom)
    
    # Create top ticker with actual news
    repeated_top = (ticker_text + "   ") * 3
    ticker_file_top = os.path.join(TMP_DIR, "ticker_top.txt")
    with open(ticker_file_top, "w", encoding="utf-8") as f:
        f.write(repeated_top)

    cmd = [
        "ffmpeg", "-y", "-i", input_video,
        "-vf", 
        # Bottom ticker (existing)
        f"drawbox=y=ih-60:color=red@1.0:width=iw:height=60:t=fill,"
        f"drawtext=fontfile={FONT_PATH}:textfile={ticker_file_bottom}:"
        f"fontsize=42:fontcolor=white:x=w-mod(t*200\\,w+tw):y=h-50:"
        f"shadowcolor=black:shadowx=2:shadowy=2,"
        # Top ticker (new)
        f"drawbox=y=0:color=red@1.0:width=iw:height=80:t=fill,"
        f"drawtext=fontfile={FONT_PATH}:textfile={ticker_file_top}:"
        f"fontsize=48:fontcolor=white:x=w-mod(t*250\\,w+tw):y=15:"
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
        
        # Pre-process all media files to get total duration needed
        all_text_media_paths = []
        total_duration_needed = 0
        text_segments_info = []
        
        for i, item in enumerate(req.content):
            if item.segment_type == "video_with_text":
                media_path = download_file(item.media_url, f"media_{i}.mp4")
                all_text_media_paths.append(media_path)
                
                # Calculate text pages for this item
                max_chars_per_page = 250
                if len(item.text) > max_chars_per_page:
                    sentences = item.text.replace('। ', '।\n').replace('. ', '.\n').split('\n')
                    text_pages = []
                    current_page = []
                    current_length = 0
                    
                    for sentence in sentences:
                        sentence = sentence.strip()
                        if not sentence:
                            continue
                            
                        if current_length + len(sentence) > max_chars_per_page and current_page:
                            text_pages.append(' '.join(current_page))
                            current_page = [sentence]
                            current_length = len(sentence)
                        else:
                            current_page.append(sentence)
                            current_length += len(sentence) + 1
                    
                    if current_page:
                        text_pages.append(' '.join(current_page))
                else:
                    text_pages = [item.text]
                
                # Store info for later use
                text_segments_info.append((i, item, media_path, text_pages))
                
                # Estimate duration
                for page_text in text_pages:
                    estimated_duration = len(page_text) * 0.06
                    total_duration_needed += estimated_duration

        # Create a single looped media file for all text segments if needed
        combined_media_path = None
        if all_text_media_paths and total_duration_needed > 0:
            combined_media_path = os.path.join(TMP_DIR, "combined_media.mp4")
            create_continuous_media_loop(all_text_media_paths, total_duration_needed, combined_media_path)

        # Process all segments
        current_media_time = 0
        for i, item in enumerate(req.content):
            if item.segment_type == "video_without_text":
                # Download and process video without text
                media_path = download_file(item.media_url, f"media_{i}.mp4")
                processed_video = os.path.join(TMP_DIR, f"processed_{i}.mp4")
                process_video_without_text(media_path, processed_video)
                segments.append(processed_video)

            elif item.segment_type == "video_with_text":
                # Find the pre-processed info for this segment
                segment_info = None
                for info in text_segments_info:
                    if info[0] == i:
                        segment_info = info
                        break
                
                if segment_info:
                    _, _, media_path, text_pages = segment_info
                    
                    # Create a segment for each text page
                    for page_idx, page_text in enumerate(text_pages):
                        tts_path = os.path.join(TMP_DIR, f"tts_{i}_{page_idx}.mp3")
                        generate_tts(page_text, req.language_name, req.language_code, tts_path)
                        duration = get_audio_duration(tts_path)

                        text_bg = os.path.join(TMP_DIR, f"textbg_{i}_{page_idx}.mp4")
                        final_clip = os.path.join(TMP_DIR, f"final_{i}_{page_idx}.mp4")

                        create_text_background(bg_path, page_text, tts_path, text_bg)
                        
                        # Use the continuous media with proper timing
                        if combined_media_path:
                            overlay_continuous_media(text_bg, combined_media_path, frame_path, 
                                                   current_media_time, duration, final_clip)
                            current_media_time += duration
                        else:
                            # Fallback to original method if no combined media
                            right_clip = os.path.join(TMP_DIR, f"right_{i}_{page_idx}.mp4")
                            extend_media(media_path, duration, right_clip)
                            overlay_right_and_frame(text_bg, right_clip, frame_path, final_clip)
                        
                        segments.append(final_clip)

        # Concatenate all segments
        merged_video = os.path.join(VIDEO_BULLETIN_DIR, f"merged_{int(time.time())}.mp4")
        concat_videos(segments, merged_video)

        # Add ticker
        final_filename = f"bulletin_{int(time.time())}.mp4"
        final_output = os.path.join(VIDEO_BULLETIN_DIR, final_filename)
        add_ticker(merged_video, req.ticker, final_output)
        
        # Generate thumbnail from the final video
        thumbnail_path = os.path.join(THUMBNAIL_DIR, f"thumb_{final_filename.replace('.mp4', '.jpg')}")
        generate_thumbnail(final_output, thumbnail_path)

        return {"status": "success", "output_path": final_output, "thumbnail_path": thumbnail_path}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def create_continuous_media_loop(media_files, total_duration, output_path):
    """Create a continuous looped video from multiple media files"""
    if not media_files:
        return
        
    # First, concatenate all media files
    concat_list = os.path.join(TMP_DIR, "concat_media.txt")
    with open(concat_list, "w") as f:
        for media_path in media_files:
            f.write(f"file '{os.path.abspath(media_path)}'\n")
    
    # Create concatenated video
    temp_concat = os.path.join(TMP_DIR, "temp_concat.mp4")
    concat_cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_list,
        "-vf", "scale=1100:620",
        "-c:v", "libx264",
        temp_concat
    ]
    subprocess.run(concat_cmd, check=True)
    
    # Loop the concatenated video for the total duration
    loop_cmd = [
        "ffmpeg", "-y",
        "-stream_loop", "-1",
        "-i", temp_concat,
        "-t", str(total_duration * 1.2),  # Add 20% buffer
        "-c:v", "libx264",
        output_path
    ]
    subprocess.run(loop_cmd, check=True)

def overlay_continuous_media(bg_text_video, continuous_media, frame_img, start_time, duration, output_path):
    """Overlay a portion of continuous media starting from a specific time"""
    cmd = [
        "ffmpeg", "-y",
        "-i", bg_text_video,
        "-ss", str(start_time),
        "-t", str(duration),
        "-i", continuous_media,
        "-i", frame_img,
        "-filter_complex",
        "[1:v]scale=1100:620[media];"
        "[0:v][media]overlay=760:220[tmp];"
        "[tmp][2:v]overlay=0:0[out]",
        "-map", "[out]", 
        "-map", "0:a",  # Keep original audio from bg_text_video
        "-c:v", "libx264", 
        "-c:a", "aac", 
        output_path
    ]
    subprocess.run(cmd, check=True)

def generate_thumbnail(video_path, thumbnail_path):
    """Generate thumbnail from video at 5 second mark"""
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-ss", "00:00:05",  # Take frame at 5 seconds
        "-vframes", "1",
        "-vf", "scale=1920:1080",
        thumbnail_path
    ]
    subprocess.run(cmd, check=True)

# === Run with uvicorn ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)