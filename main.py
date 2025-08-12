import os
import subprocess
import textwrap
import requests
import time
import re
import unicodedata
import uuid
import shutil
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from moviepy.editor import VideoFileClip, ImageClip, concatenate_videoclips
from mutagen.mp3 import MP3
from google.cloud import texttospeech
import urllib3

# Suppress SSL warnings for development (remove in production)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

app = FastAPI()

# Setup paths
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key.json"
OUTPUT_DIR = "output"
VIDEO_BULLETIN_DIR = "video_bulletin"
THUMBNAIL_DIR = "thumbnail"
FONT_PATH = "assets/NotoSans-Regular.ttf"
DEFAULT_BG_PATH = "assets/default_background.mp4"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIDEO_BULLETIN_DIR, exist_ok=True)
os.makedirs(THUMBNAIL_DIR, exist_ok=True)
os.makedirs(os.path.dirname(DEFAULT_BG_PATH), exist_ok=True)

# Request models
class ContentItem(BaseModel):
    segment_type: str
    media_url: str
    text: Optional[str] = None
    top_headline: Optional[str] = None
    frame_url: Optional[str] = None

class BulletinData(BaseModel):
    id: str
    logo_url: Optional[str] = None
    language_code: str
    language_name: str
    ticker: str
    background_url: str
    story_thumbnail: Optional[str] = None
    generated_story_url: Optional[str] = None
    content: List[ContentItem]

class BulletinResponse(BaseModel):
    status: bool
    message: str
    data: List[BulletinData]

# === Hindi Text Correction for Google TTS ===
def correct_hindi_text(text):
    """Correct Hindi text for better Google TTS pronunciation"""
    if not text:
        return text
    
    text = unicodedata.normalize('NFC', text)
    
    hindi_corrections = {
        'मै': 'मैं',
        'मे': 'में',
        'हे': 'है',
        'कया': 'क्या',
        'कयों': 'क्यों',
        'बहत': 'बहुत',
        'अभि': 'अभी',
        'कभि': 'कभी',
        'सभि': 'सभी',
        'यहां': 'यहाँ',
        'वहां': 'वहाँ',
        'कहां': 'कहाँ',
        'जहां': 'जहाँ',
        'तुमहारा': 'तुम्हारा',
        'तुमहारे': 'तुम्हारे',
        'तुमहारी': 'तुम्हारी',
        'उन्हे': 'उन्हें',
        'इन्हे': 'इन्हें',
        'गये': 'गए',
        'गयी': 'गई',
        'आये': 'आए',
        'आयी': 'आई',
        'हुये': 'हुए',
        'हुयी': 'हुई',
        'किये': 'किए',
        'दिये': 'दिए',
        'लिये': 'लिए',
        'चाहिये': 'चाहिए',
        'नये': 'नए',
        'नयी': 'नई',
        'अछा': 'अच्छा',
        'अछे': 'अच्छे',
        'अछी': 'अच्छी',
        'बडा': 'बड़ा',
        'बडे': 'बड़े',
        'बडी': 'बड़ी',
        'आदमि': 'आदमी',
        'लडका': 'लड़का',
        'लडकी': 'लड़की',
        'पानि': 'पानी',
        'गांव': 'गाँव',
        'पांच': 'पाँच',
        'छः': 'छह',
        'हजार': 'हज़ार',
        'करोड': 'करोड़',
        'के लिये': 'के लिए',
        'की लिये': 'के लिए',
        'इसलिये': 'इसलिए',
    }
    
    for wrong, correct in hindi_corrections.items():
        pattern = r'\b' + re.escape(wrong) + r'\b'
        text = re.sub(pattern, correct, text)
    
    compound_fixes = [
        (r'\bहो\s+गये\b', 'हो गए'),
        (r'\bहो\s+गयी\b', 'हो गई'),
        (r'\bहो\s+गया\b', 'हो गया'),
        (r'\bचला\s+गया\b', 'चला गया'),
        (r'\bचली\s+गयी\b', 'चली गई'),
        (r'\bचले\s+गये\b', 'चले गए'),
        (r'\bआ\s+गया\b', 'आ गया'),
        (r'\bआ\s+गयी\b', 'आ गई'),
        (r'\bआ\s+गये\b', 'आ गए'),
        (r'\bकर\s+दिया\b', 'कर दिया'),
        (r'\bकर\s+दिये\b', 'कर दिए'),
        (r'\bले\s+लिया\b', 'ले लिया'),
        (r'\bले\s+लिये\b', 'ले लिए'),
    ]
    
    for pattern, replacement in compound_fixes:
        text = re.sub(pattern, replacement, text)
    
    if text.strip() and not text.strip().endswith(('।', '.', '!', '?')):
        text = text.strip() + '।'
    
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def wrap_hindi_text(text, max_chars_per_line=25):
    """Simple text wrapping for Hindi"""
    corrected_text = correct_hindi_text(text)
    words = corrected_text.split()
    lines = []
    current_line = ""
    
    for word in words:
        if current_line and len(current_line + " " + word) > max_chars_per_line:
            lines.append(current_line.strip())
            current_line = word
        else:
            if current_line:
                current_line += " " + word
            else:
                current_line = word
    
    if current_line.strip():
        lines.append(current_line.strip())
    
    return '\n'.join(lines)

def split_text_into_pages(text, max_chars_per_page=150):
    """Split text into pages for TTS with byte-aware splitting"""
    corrected_text = correct_hindi_text(text)
    
    sentence_endings = ['।', '.', '!', '?']
    sentences = []
    current_sentence = ""
    
    for char in corrected_text:
        current_sentence += char
        if char in sentence_endings:
            sentence = current_sentence.strip()
            if sentence:
                byte_length = len(sentence.encode('utf-8'))
                if byte_length > 850:
                    words = sentence.split()
                    temp_sentence = ""
                    for word in words:
                        test_sentence = (temp_sentence + " " + word).strip() if temp_sentence else word
                        if len(test_sentence.encode('utf-8')) > 850:
                            if temp_sentence:
                                sentences.append(temp_sentence.strip() + '।')
                            temp_sentence = word
                        else:
                            temp_sentence = test_sentence
                    if temp_sentence:
                        sentences.append(temp_sentence.strip() + ('।' if not temp_sentence.strip().endswith(tuple(sentence_endings)) else ''))
                else:
                    sentences.append(sentence)
            current_sentence = ""
    
    if current_sentence.strip():
        remaining = current_sentence.strip()
        byte_length = len(remaining.encode('utf-8'))
        if byte_length > 850:
            words = remaining.split()
            temp_sentence = ""
            for word in words:
                test_sentence = (temp_sentence + " " + word).strip() if temp_sentence else word
                if len(test_sentence.encode('utf-8')) > 850:
                    if temp_sentence:
                        sentences.append(temp_sentence.strip() + '।')
                    temp_sentence = word
                else:
                    temp_sentence = test_sentence
            if temp_sentence:
                sentences.append(temp_sentence.strip() + '।')
        else:
            sentences.append(remaining + ('।' if not remaining.endswith(tuple(sentence_endings)) else ''))
    
    pages = []
    current_page = []
    current_page_bytes = 0
    max_bytes_per_page = 450
    
    for sentence in sentences:
        sentence_bytes = len(sentence.encode('utf-8'))
        
        if current_page and (current_page_bytes + sentence_bytes + 1) > max_bytes_per_page:
            pages.append(' '.join(current_page))
            current_page = [sentence]
            current_page_bytes = sentence_bytes
        else:
            current_page.append(sentence)
            current_page_bytes += sentence_bytes + (1 if current_page else 0)
    
    if current_page:
        pages.append(' '.join(current_page))
    
    validated_pages = []
    for page in pages:
        page_bytes = len(page.encode('utf-8'))
        if page_bytes > 900:
            words = page.split()
            chunk = ""
            for word in words:
                test_chunk = (chunk + " " + word).strip() if chunk else word
                if len(test_chunk.encode('utf-8')) > 850:
                    if chunk:
                        validated_pages.append(chunk.strip())
                    chunk = word
                else:
                    chunk = test_chunk
            if chunk:
                validated_pages.append(chunk.strip())
        else:
            validated_pages.append(page)
    
    return validated_pages if validated_pages else [corrected_text[:100] + '।']

def process_content_text(text, max_chars_per_page=150):
    """Process content text for TTS with proper byte handling"""
    if not text:
        return []
    
    if len(text.encode('utf-8')) > 900:
        return split_text_into_pages(text, max_chars_per_page)
    else:
        corrected = correct_hindi_text(text)
        return [corrected]

# === Helper functions ===
def download_file(url, filename, temp_dir, max_retries=3, backoff_factor=2):
    """Download file with retry logic and exponential backoff"""
    path = os.path.join(temp_dir, filename)
    
    for attempt in range(max_retries):
        try:
            print(f"🔄 Downloading: {url} (Attempt {attempt + 1}/{max_retries})")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'video/mp4,video/*,*/*',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive'
            }
            
            r = requests.get(url, headers=headers, timeout=60, stream=True, verify=False)
            r.raise_for_status()
            
            if r.status_code == 200:
                total_size = int(r.headers.get('content-length', 0))
                with open(path, "wb") as f:
                    downloaded = 0
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                percent = (downloaded / total_size) * 100
                                print(f"📥 Downloaded: {percent:.1f}%", end='\r')
                
                print(f"✅ Downloaded: {filename} ({os.path.getsize(path)} bytes)")
                
                if os.path.getsize(path) == 0:
                    raise Exception(f"Downloaded file is empty: {filename}")
                
                return path
                
        except (requests.exceptions.RequestException, Exception) as e:
            print(f"⚠️ Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                sleep_time = backoff_factor ** attempt
                print(f"⏳ Retrying after {sleep_time} seconds...")
                time.sleep(sleep_time)
                continue
            raise Exception(f"Download failed after {max_retries} attempts: {str(e)}")

def generate_tts(text, voice_name, lang_code, output_path):
    """Generate TTS with corrected Hindi text"""
    corrected_text = correct_hindi_text(text)
    
    text_bytes = len(corrected_text.encode('utf-8'))
    if text_bytes > 900:
        print(f"⚠️ Text too long ({text_bytes} bytes), truncating...")
        truncated = corrected_text
        while len(truncated.encode('utf-8')) > 850:
            truncated = truncated[:-10]
        truncated = truncated.strip() + '।'
        corrected_text = truncated
    
    try:
        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=corrected_text)
        voice = texttospeech.VoiceSelectionParams(language_code=lang_code, name=voice_name)
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
        response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
        
        with open(output_path, "wb") as out:
            out.write(response.audio_content)
            
    except Exception as e:
        print(f"❌ TTS generation failed: {str(e)}")
        raise

def get_audio_duration(audio_path):
    """Get audio duration"""
    try:
        return MP3(audio_path).info.length
    except:
        return 5.0

def get_video_duration(video_path):
    """Get video duration safely"""
    try:
        cmd = [
            "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except:
        pass
    return 10.0

def run_ffmpeg_command(cmd, description="FFmpeg command", timeout=60):
    """Run FFmpeg command with error handling"""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            print(f"⚠️ {description} failed with code {result.returncode}")
            print(f"Error: {result.stderr[:500]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"⚠️ {description} timed out")
        return False
    except Exception as e:
        print(f"⚠️ {description} exception: {e}")
        return False

def get_compatible_encoding_params():
    """Get FFmpeg encoding parameters for maximum compatibility"""
    return [
        "-c:v", "libx264",
        "-profile:v", "baseline",
        "-level", "3.0",
        "-pix_fmt", "yuv420p",
        "-preset", "slow",
        "-crf", "23",
        "-movflags", "+faststart",
        "-max_muxing_queue_size", "9999"
    ]

def get_audio_encoding_params():
    """Get audio encoding parameters for compatibility"""
    return [
        "-c:a", "aac",
        "-b:a", "128k",
        "-ar", "44100",
        "-ac", "2"
    ]

def create_dynamic_text_segments(bg_path, text_pages, audio_paths, media_path, frame_path, 
                                output_path, temp_dir, headlines=None):
    """Create a single video with dynamic text that changes in the black box"""
    
    if not headlines:
        headlines = ["ब्रेकिंग न्यूज़"] * len(text_pages)
    
    font_path = os.path.abspath(FONT_PATH)
    
    # Calculate total duration
    total_duration = sum(get_audio_duration(audio_path) for audio_path in audio_paths)
    
    # Create looped media video for the right side
    temp_media_looped = os.path.join(temp_dir, "media_looped.mp4")
    media_loop_cmd = [
        "ffmpeg", "-y",
        "-stream_loop", "-1",
        "-i", media_path,
        "-t", str(total_duration),
        "-vf", "scale=1100:620:force_original_aspect_ratio=decrease,pad=1100:620:(ow-iw)/2:(oh-ih)/2,format=yuv420p",
        "-r", "30"
    ] + get_compatible_encoding_params() + [temp_media_looped]
    
    if not run_ffmpeg_command(media_loop_cmd, "Loop media video"):
        shutil.copy2(media_path, temp_media_looped)
    
    # Create looped background
    temp_bg_looped = os.path.join(temp_dir, "bg_looped.mp4")
    bg_loop_cmd = [
        "ffmpeg", "-y",
        "-stream_loop", "-1",
        "-i", bg_path,
        "-t", str(total_duration)
    ] + get_compatible_encoding_params() + [temp_bg_looped]
    
    if not run_ffmpeg_command(bg_loop_cmd, "Loop background"):
        shutil.copy2(bg_path, temp_bg_looped)
    
    # Concatenate all audio files
    concat_audio_file = os.path.join(temp_dir, "concat_audio.txt")
    with open(concat_audio_file, "w") as f:
        for audio_path in audio_paths:
            f.write(f"file '{os.path.abspath(audio_path)}'\n")
    
    combined_audio = os.path.join(temp_dir, "combined_audio.mp3")
    audio_concat_cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_audio_file,
        "-c", "copy",
        combined_audio
    ]
    
    if not run_ffmpeg_command(audio_concat_cmd, "Concatenate audio"):
        combined_audio = audio_paths[0]  # Fallback to first audio
    
    # Create subtitle file with timing for each text segment
    srt_file = os.path.join(temp_dir, "subtitles.srt")
    current_time = 0
    
    with open(srt_file, "w", encoding="utf-8") as f:
        for i, (text, audio_path, headline) in enumerate(zip(text_pages, audio_paths, headlines)):
            duration = get_audio_duration(audio_path)
            start_time = current_time
            end_time = current_time + duration
            
            # Format time for SRT
            def format_srt_time(seconds):
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                secs = seconds % 60
                return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace('.', ',')
            
            # Write SRT entry
            f.write(f"{i + 1}\n")
            f.write(f"{format_srt_time(start_time)} --> {format_srt_time(end_time)}\n")
            
            # Format text with headline
            wrapped_text = wrap_hindi_text(text, max_chars_per_line=25)
            # Add special formatting tags for headline
            f.write(f"{{\\an7}}{{\\fs18}}{{\\c&H0000FF&}}{headline}{{\\r}}\n")
            f.write(f"{wrapped_text}\n\n")
            
            current_time = end_time
    
    # Build the complex filter for dynamic text
    box_width = 650
    box_height = 460
    box_y = 220
    
    # Create the filter complex with dynamic text using subtitles
    filter_str = f"""
    [0:v]format=yuv420p[bg];
    [1:v]format=yuv420p[media];
    [bg][media]overlay=760:220:shortest=1[with_media];
    [with_media]
    drawbox=x=40:y={box_y}:w={box_width}:h={box_height}:color=black@0.7:t=fill,
    drawbox=x=40:y={box_y}:w={box_width}:h=50:color=red@1.0:t=fill,
    subtitles={srt_file}:force_style='FontName={os.path.basename(FONT_PATH)},FontSize=20,PrimaryColour=&HFFFFFF&,OutlineColour=&H000000&,Outline=2,Alignment=7,MarginL=60,MarginV={box_y + 70}',
    format=yuv420p[with_text]
    """
    
    # Add frame overlay if available
    if frame_path and os.path.exists(frame_path):
        filter_str += ";[with_text][2:v]overlay=0:0,format=yuv420p[out]"
        map_str = "[out]"
        inputs = ["-i", temp_bg_looped, "-i", temp_media_looped, "-i", frame_path, "-i", combined_audio]
    else:
        filter_str = filter_str.replace("[with_text]", "[out]")
        map_str = "[out]"
        inputs = ["-i", temp_bg_looped, "-i", temp_media_looped, "-i", combined_audio]
    
    # Create the final command
    cmd = ["ffmpeg", "-y"] + inputs + [
        "-filter_complex", filter_str,
        "-map", map_str,
        "-map", f"{len(inputs) - 1}:a"
    ] + get_compatible_encoding_params() + get_audio_encoding_params() + [output_path]
    
    if not run_ffmpeg_command(cmd, "Create dynamic text video", timeout=120):
        print("⚠️ Falling back to alternative method...")
        # Fallback: Create individual segments and concatenate
        create_fallback_dynamic_segments(bg_path, text_pages, audio_paths, media_path, 
                                        frame_path, output_path, temp_dir, headlines)

def create_fallback_dynamic_segments(bg_path, text_pages, audio_paths, media_path, 
                                    frame_path, output_path, temp_dir, headlines):
    """Fallback method: Create individual segments with text and concatenate them"""
    
    segment_files = []
    media_offset = 0
    
    for i, (text, audio_path, headline) in enumerate(zip(text_pages, audio_paths, headlines)):
        segment_output = os.path.join(temp_dir, f"text_segment_{i}.mp4")
        
        # Create text background for this segment
        text_bg = os.path.join(temp_dir, f"text_bg_{i}.mp4")
        create_text_background(bg_path, text, audio_path, text_bg, temp_dir, headline)
        
        # Get duration for media segment
        duration = get_audio_duration(audio_path)
        
        # Extract media segment
        media_segment = os.path.join(temp_dir, f"media_seg_{i}.mp4")
        media_cmd = [
            "ffmpeg", "-y",
            "-stream_loop", "-1",
            "-i", media_path,
            "-ss", str(media_offset),
            "-t", str(duration),
            "-vf", "scale=1100:620:force_original_aspect_ratio=decrease,pad=1100:620:(ow-iw)/2:(oh-ih)/2,format=yuv420p"
        ] + get_compatible_encoding_params() + [media_segment]
        
        if not run_ffmpeg_command(media_cmd, f"Extract media segment {i}"):
            media_segment = media_path
        
        # Overlay media on text background
        if frame_path and os.path.exists(frame_path):
            overlay_cmd = [
                "ffmpeg", "-y",
                "-i", text_bg,
                "-i", media_segment,
                "-i", frame_path,
                "-filter_complex",
                "[0:v][1:v]overlay=760:220:shortest=1[tmp];"
                "[tmp][2:v]overlay=0:0,format=yuv420p[out]",
                "-map", "[out]",
                "-map", "0:a?"
            ] + get_compatible_encoding_params() + get_audio_encoding_params() + [segment_output]
        else:
            overlay_cmd = [
                "ffmpeg", "-y",
                "-i", text_bg,
                "-i", media_segment,
                "-filter_complex",
                "[0:v][1:v]overlay=760:220:shortest=1,format=yuv420p[out]",
                "-map", "[out]",
                "-map", "0:a?"
            ] + get_compatible_encoding_params() + get_audio_encoding_params() + [segment_output]
        
        if run_ffmpeg_command(overlay_cmd, f"Create segment {i}"):
            segment_files.append(segment_output)
        
        media_offset += duration
    
    # Concatenate all segments
    if segment_files:
        concat_file = os.path.join(temp_dir, "concat_segments.txt")
        with open(concat_file, "w") as f:
            for segment in segment_files:
                f.write(f"file '{os.path.abspath(segment)}'\n")
        
        concat_cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file
        ] + get_compatible_encoding_params() + get_audio_encoding_params() + [output_path]
        
        run_ffmpeg_command(concat_cmd, "Concatenate text segments")

def create_text_background(bg_path, text, audio_path, output_path, temp_dir, headline=None):
    """Create background video with text overlay and TTS audio"""
    font_path = os.path.abspath(FONT_PATH)
    subtitle_file = os.path.join(temp_dir, "tts_text.txt")
    breaking_news_file = os.path.join(temp_dir, "breaking_news.txt")

    box_width = 650
    box_height = 460
    box_y = 220
    header_y = box_y + 10
    text_y = box_y + 70
    
    header_text = headline if headline else "ब्रेकिंग न्यूज़"
    
    with open(breaking_news_file, "w", encoding="utf-8") as f:
        f.write(correct_hindi_text(header_text))
    
    wrapped_text = wrap_hindi_text(text, max_chars_per_line=25)
    
    with open(subtitle_file, "w", encoding="utf-8") as f:
        f.write(wrapped_text)

    audio_duration = get_audio_duration(audio_path)
    
    # Create looped background
    temp_bg_looped = os.path.join(temp_dir, "bg_looped_temp.mp4")
    loop_cmd = [
        "ffmpeg", "-y",
        "-stream_loop", "-1",
        "-i", bg_path,
        "-t", str(audio_duration)
    ] + get_compatible_encoding_params() + [temp_bg_looped]
    
    if not run_ffmpeg_command(loop_cmd, "Background loop"):
        shutil.copy2(bg_path, temp_bg_looped)

    # Create filter with text overlay
    filter_str = (
        f"[0:v]format=yuv420p,"
        f"drawbox=x=40:y={box_y}:w={box_width}:h={box_height}:color=black@0.7:t=fill,"
        f"drawbox=x=40:y={box_y}:w={box_width}:h=50:color=red@1.0:t=fill,"
        f"drawtext=fontfile={font_path}:textfile={breaking_news_file}:"
        f"fontcolor=white:fontsize=36:x=(40+{box_width}/2-text_w/2):y={header_y}:"
        f"shadowcolor=black:shadowx=2:shadowy=2,"
        f"drawtext=fontfile={font_path}:textfile={subtitle_file}:"
        f"fontcolor=white:fontsize=46:x=60:y={text_y}:"
        f"shadowcolor=black:shadowx=2:shadowy=2,format=yuv420p[v]"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", temp_bg_looped,
        "-i", audio_path,
        "-filter_complex", filter_str,
        "-map", "[v]",
        "-map", "1:a"
    ] + get_compatible_encoding_params() + get_audio_encoding_params() + [output_path]
    
    if not run_ffmpeg_command(cmd, "Text overlay"):
        # Fallback with simpler encoding
        fallback_cmd = [
            "ffmpeg", "-y",
            "-i", temp_bg_looped,
            "-i", audio_path,
            "-map", "0:v",
            "-map", "1:a",
            "-shortest"
        ] + get_compatible_encoding_params() + get_audio_encoding_params() + [output_path]
        run_ffmpeg_command(fallback_cmd, "Fallback text background")

def process_video_without_text(media_path, output_path):
    """Process video without text with maximum compatibility"""
    try:
        probe_cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=index",
            "-of", "csv=p=0",
            media_path
        ]
        
        result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=5)
        has_audio = bool(result.stdout.strip())
        
        print(f"📹 Processing video without text (has audio: {has_audio})")
        
        if has_audio:
            cmd = [
                "ffmpeg", "-y",
                "-i", media_path,
                "-vf", "format=yuv420p"
            ] + get_compatible_encoding_params() + get_audio_encoding_params() + [output_path]
        else:
            video_duration = get_video_duration(media_path)
            cmd = [
                "ffmpeg", "-y",
                "-i", media_path,
                "-f", "lavfi",
                "-i", f"anullsrc=channel_layout=stereo:sample_rate=44100:duration={video_duration}",
                "-vf", "format=yuv420p",
                "-map", "0:v",
                "-map", "1:a",
                "-shortest"
            ] + get_compatible_encoding_params() + get_audio_encoding_params() + [output_path]
        
        if not run_ffmpeg_command(cmd, "Process video without text", timeout=90):
            fallback_cmd = [
                "ffmpeg", "-y",
                "-i", media_path,
                "-vf", "format=yuv420p"
            ] + get_compatible_encoding_params() + [output_path]
            
            if not run_ffmpeg_command(fallback_cmd, "Fallback video processing"):
                shutil.copy2(media_path, output_path)
        
    except Exception as e:
        print(f"❌ Error processing video without text: {e}")
        shutil.copy2(media_path, output_path)

def concat_videos(video_paths, output_path):
    """Concatenate videos with maximum compatibility"""
    if not video_paths:
        raise Exception("No videos to concatenate")
    
    normalized_videos = []
    temp_dir = os.path.dirname(video_paths[0])
    
    for i, video_path in enumerate(video_paths):
        if not os.path.exists(video_path):
            continue
            
        normalized_path = os.path.join(temp_dir, f"norm_concat_{i}.mp4")
        
        probe_cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=index",
            "-of", "csv=p=0",
            video_path
        ]
        
        result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=5)
        has_audio = bool(result.stdout.strip())
        
        if has_audio:
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
                "-r", "30"
            ] + get_compatible_encoding_params() + get_audio_encoding_params() + [normalized_path]
        else:
            duration = get_video_duration(video_path)
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-f", "lavfi",
                "-i", f"anullsrc=channel_layout=stereo:sample_rate=44100:duration={duration}",
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
                "-r", "30",
                "-map", "0:v",
                "-map", "1:a",
                "-shortest"
            ] + get_compatible_encoding_params() + get_audio_encoding_params() + [normalized_path]
        
        if run_ffmpeg_command(cmd, f"Normalize video {i}"):
            normalized_videos.append(normalized_path)
        else:
            normalized_videos.append(video_path)
    
    if not normalized_videos:
        raise Exception("No videos could be normalized")
    
    concat_file = os.path.join(temp_dir, "concat_list.txt")
    with open(concat_file, "w") as f:
        for video in normalized_videos:
            f.write(f"file '{os.path.abspath(video)}'\n")
    
    concat_cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_file,
        "-vf", "format=yuv420p"
    ] + get_compatible_encoding_params() + get_audio_encoding_params() + [output_path]
    
    if not run_ffmpeg_command(concat_cmd, "Concatenate videos", timeout=120):
        print("⚠️ Concat failed, using first video only")
        fallback_cmd = [
            "ffmpeg", "-y",
            "-i", normalized_videos[0]
        ] + get_compatible_encoding_params() + get_audio_encoding_params() + [output_path]
        
        if not run_ffmpeg_command(fallback_cmd, "Fallback concat"):
            shutil.copy2(normalized_videos[0], output_path)
    
    for video in normalized_videos:
        if video not in video_paths and os.path.exists(video):
            try:
                os.remove(video)
            except:
                pass

def add_ticker_to_segment(input_video, ticker_text, output_path, temp_dir):
    """Add ticker overlay with compatibility"""
    fixed_ticker = correct_hindi_text(ticker_text)
    
    repeated_bottom = (fixed_ticker + "   ") * 3
    ticker_file_bottom = os.path.join(temp_dir, "ticker_bottom.txt")
    with open(ticker_file_bottom, "w", encoding="utf-8") as f:
        f.write(repeated_bottom)
    
    ticker_file_top = os.path.join(temp_dir, "ticker_top.txt")
    with open(ticker_file_top, "w", encoding="utf-8") as f:
        f.write(fixed_ticker)

    cmd = [
        "ffmpeg", "-y", "-i", input_video,
        "-vf", 
        f"format=yuv420p,"
        f"drawbox=y=ih-60:color=red@1.0:width=iw:height=60:t=fill,"
        f"drawtext=fontfile={FONT_PATH}:textfile={ticker_file_bottom}:"
        f"fontsize=42:fontcolor=white:x=w-mod(t*200\\,w+tw):y=h-50:"
        f"shadowcolor=black:shadowx=2:shadowy=2,"
        f"drawbox=y=0:color=red@1.0:width=iw:height=80:t=fill,"
        f"drawtext=fontfile={FONT_PATH}:textfile={ticker_file_top}:"
        f"fontsize=48:fontcolor=white:x=(w-text_w)/2:y=15:"
        f"shadowcolor=black:shadowx=2:shadowy=2,"
        f"format=yuv420p"
    ] + get_compatible_encoding_params() + [
        "-c:a", "copy",
        output_path
    ]

    if not run_ffmpeg_command(cmd, "Add ticker"):
        fallback_cmd = [
            "ffmpeg", "-y",
            "-i", input_video
        ] + get_compatible_encoding_params() + [
            "-c:a", "copy",
            output_path
        ]
        if not run_ffmpeg_command(fallback_cmd, "Ticker fallback"):
            shutil.copy2(input_video, output_path)

def generate_thumbnail(video_path, thumbnail_path):
    """Generate thumbnail from video"""
    try:
        duration = get_video_duration(video_path)
        seek_time = min(2.0, duration / 2)
        
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(seek_time),
            "-i", video_path,
            "-vframes", "1",
            "-vf", "scale=1920:1080",
            "-q:v", "2",
            thumbnail_path
        ]
        
        if not run_ffmpeg_command(cmd, "Generate thumbnail"):
            fallback_cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-vframes", "1",
                "-vf", "scale=1920:1080",
                thumbnail_path
            ]
            run_ffmpeg_command(fallback_cmd, "Thumbnail fallback")
    except Exception as e:
        print(f"❌ Thumbnail generation failed: {e}")

def cleanup_request_files(request_temp_dir):
    """Clean up temporary files"""
    try:
        if os.path.exists(request_temp_dir):
            shutil.rmtree(request_temp_dir)
            print(f"✅ Cleaned up: {request_temp_dir}")
    except Exception as e:
        print(f"⚠️ Cleanup failed: {e}")

frame_cache = {}

def get_or_download_frame(frame_url, request_temp_dir, segment_id):
    """Download and cache frame files"""
    if not frame_url:
        return None
    
    if frame_url in frame_cache:
        return frame_cache[frame_url]
    
    try:
        frame_filename = f"frame_{segment_id}.png"
        frame_path = download_file(frame_url, frame_filename, request_temp_dir)
        frame_cache[frame_url] = frame_path
        return frame_path
    except Exception as e:
        print(f"⚠️ Failed to download frame: {e}")
        return None

def generate_bulletin_from_data(bulletin_data: BulletinData):
    """Main bulletin generation function with dynamic text updates"""
    request_id = str(uuid.uuid4())
    base_temp_dir = "temp"
    request_temp_dir = os.path.join(base_temp_dir, f"request_{request_id}")
    
    global frame_cache
    frame_cache = {}
    
    try:
        os.makedirs(base_temp_dir, exist_ok=True)
        os.makedirs(request_temp_dir, exist_ok=True)
        
        print(f"🚀 Starting bulletin generation with dynamic text")
        print(f"📁 Request ID: {request_id}")
        print(f"🆔 Bulletin ID: {bulletin_data.id}")
        
        # Download background with fallback
        bg_path = None
        try:
            bg_path = download_file(bulletin_data.background_url, "bg.mp4", request_temp_dir)
        except Exception as e:
            print(f"⚠️ Failed to download background: {str(e)}")
            if os.path.exists(DEFAULT_BG_PATH):
                print(f"📌 Using default background: {DEFAULT_BG_PATH}")
                bg_path = DEFAULT_BG_PATH
            else:
                raise HTTPException(status_code=400, detail=f"No background available: {str(e)}")
        
        segments = []
        
        # Process each content item
        for i, item in enumerate(bulletin_data.content):
            if item.segment_type == "video_without_text":
                try:
                    media_path = download_file(item.media_url, f"media_{i}.mp4", request_temp_dir)
                    processed_video = os.path.join(request_temp_dir, f"processed_{i}.mp4")
                    process_video_without_text(media_path, processed_video)
                    segments.append(processed_video)
                    print(f"✅ Added video_without_text segment {i}")
                except Exception as e:
                    print(f"⚠️ Failed to process video_without_text {i}: {e}")
                    continue

            elif item.segment_type == "video_with_text" and item.text:
                try:
                    # Download media for right side
                    media_path = download_file(item.media_url, f"media_{i}.mp4", request_temp_dir)
                    
                    # Get frame if available
                    frame_path = get_or_download_frame(item.frame_url, request_temp_dir, i)
                    
                    # Split text into pages
                    text_pages = process_content_text(item.text, max_chars_per_page=150)
                    
                    if text_pages:
                        # Generate TTS for each text page
                        audio_paths = []
                        headlines = []
                        
                        for page_idx, page_text in enumerate(text_pages):
                            tts_path = os.path.join(request_temp_dir, f"tts_{i}_{page_idx}.mp3")
                            generate_tts(page_text, bulletin_data.language_name, 
                                       bulletin_data.language_code, tts_path)
                            audio_paths.append(tts_path)
                            headlines.append(item.top_headline if item.top_headline else "ब्रेकिंग न्यूज़")
                        
                        # Create single video with dynamic text changes
                        dynamic_output = os.path.join(request_temp_dir, f"dynamic_{i}.mp4")
                        create_dynamic_text_segments(bg_path, text_pages, audio_paths, 
                                                    media_path, frame_path, dynamic_output, 
                                                    request_temp_dir, headlines)
                        
                        # Add ticker to the dynamic video
                        final_output = os.path.join(request_temp_dir, f"final_{i}.mp4")
                        add_ticker_to_segment(dynamic_output, bulletin_data.ticker, 
                                            final_output, request_temp_dir)
                        
                        segments.append(final_output)
                        print(f"✅ Created dynamic text segment {i} with {len(text_pages)} text changes")
                    
                except Exception as e:
                    print(f"⚠️ Failed to process video_with_text {i}: {e}")
                    continue

        if not segments:
            raise HTTPException(status_code=400, detail="No segments were generated")

        print("🔗 Concatenating segments...")
        temp_concat_output = os.path.join(request_temp_dir, "temp_final.mp4")
        concat_videos(segments, temp_concat_output)
        
        final_filename = f"bulletin_{bulletin_data.id}_{int(time.time())}.mp4"
        final_output = os.path.join(VIDEO_BULLETIN_DIR, final_filename)
        
        print("📤 Creating final output...")
        
        final_cmd = [
            "ffmpeg", "-y",
            "-i", temp_concat_output,
            "-vf", "format=yuv420p",
            "-codec:v", "libx264",
            "-profile:v", "high",
            "-level", "4.0",
            "-preset", "medium",
            "-crf", "23",
            "-codec:a", "aac",
            "-b:a", "192k",
            "-ar", "48000",
            "-ac", "2",
            "-movflags", "+faststart",
            "-metadata:s:v:0", "language=und",
            "-metadata:s:a:0", "language=hin",
            final_output
        ]
        
        if not run_ffmpeg_command(final_cmd, "Final encoding", timeout=180):
            print("⚠️ Using fallback for final output")
            shutil.copy2(temp_concat_output, final_output)
        
        if not os.path.exists(final_output):
            raise HTTPException(status_code=500, detail="Failed to create final video")

        print("🖼️ Generating thumbnail...")
        thumbnail_filename = f"thumb_{bulletin_data.id}_{int(time.time())}.jpg"
        thumbnail_path = os.path.join(THUMBNAIL_DIR, thumbnail_filename)
        generate_thumbnail(final_output, thumbnail_path)

        cleanup_request_files(request_temp_dir)

        print("🎉 Bulletin generation completed with dynamic text!")
        
        return {
            "status": "success",
            "bulletin_id": bulletin_data.id,
            "request_id": request_id,
            "output_path": final_output,
            "thumbnail_path": thumbnail_path,
            "message": f"Bulletin {bulletin_data.id} generated successfully with dynamic text",
            "video_info": {
                "codec": "H.264",
                "profile": "High",
                "pixel_format": "yuv420p",
                "audio_codec": "AAC",
                "features": "Dynamic text updates in black box"
            }
        }

    except Exception as e:
        cleanup_request_files(request_temp_dir)
        print(f"💥 Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate")
def generate_bulletin(request_data: Dict[Any, Any]):
    """Main API endpoint"""
    if "status" in request_data and "data" in request_data:
        response = BulletinResponse(**request_data)
        results = []
        
        if not response.status:
            raise HTTPException(status_code=400, detail="Response status is false")
        
        for bulletin_data in response.data:
            try:
                result = generate_bulletin_from_data(bulletin_data)
                results.append(result)
            except Exception as e:
                print(f"❌ Failed to process bulletin {bulletin_data.id}: {str(e)}")
                results.append({
                    "status": "failed",
                    "bulletin_id": bulletin_data.id,
                    "error": str(e)
                })
        
        return {
            "status": response.status,
            "message": f"Processed {len(results)} bulletins",
            "results": results
        }
    else:
        bulletin_data = BulletinData(**request_data)
        return generate_bulletin_from_data(bulletin_data)

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "bulletin-generator", "version": "2.0-dynamic"}

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Enhanced Bulletin Generator Service with Dynamic Text...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)