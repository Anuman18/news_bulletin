import os
import subprocess
import textwrap
import json
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import VideoFileClip, ImageClip, concatenate_videoclips
from mutagen.mp3 import MP3
from google.cloud import texttospeech
from pydub import AudioSegment
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import requests
from uuid import uuid4
import glob
import shutil

app = FastAPI()

# Configuration
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key.json"
FONT_PATH = "assets/hindi-bold.ttf"
OUTPUT_DIR = "output"
UPLOADS_DIR = "uploads"
VIDEO_DIR = os.path.join(UPLOADS_DIR, "video")
THUMBNAIL_DIR = os.path.join(UPLOADS_DIR, "thumbnail")

# Create directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(THUMBNAIL_DIR, exist_ok=True)

# Pydantic models for JSON input
class ContentItem(BaseModel):
    segment_type: str
    media_url: str
    text: Optional[str] = None
    top_headline: Optional[str] = None

class ContentInput(BaseModel):
    language_code: str
    language_name: str
    ticker: str
    logo_url: Optional[str] = None
    frame_url: Optional[str] = None
    background_url: Optional[str] = None
    content: List[ContentItem]

def download_file(url, output_dir, prefix="media"):
    """Download a file from a URL and save it with a unique filename."""
    try:
        ext = os.path.splitext(url)[-1] or ".mp4"
        filename = os.path.join(output_dir, f"{prefix}_{uuid4().hex}{ext}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(filename, "wb") as f:
            f.write(response.content)
        if not os.path.exists(filename) or os.path.getsize(filename) == 0:
            raise Exception(f"Downloaded file is empty: {filename}")
        return filename
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download file from {url}: {str(e)}")

def create_bulletin_thumbnail(content, logo_path):
    """Create an attractive thumbnail for the bulletin video"""
    print("🖼️  Creating bulletin thumbnail...")
    
    try:
        thumb_width = 1280
        thumb_height = 720
        thumbnail = Image.new('RGB', (thumb_width, thumb_height), color='#000033')
        draw = ImageDraw.Draw(thumbnail)
        
        for y in range(thumb_height):
            color_value = int(51 + (y / thumb_height) * 100)
            color = (0, 0, color_value)
            draw.line([(0, y), (thumb_width, y)], fill=color)
        
        header_height = 80
        draw.rectangle([0, 0, thumb_width, header_height], fill='#CC0000')
        
        try:
            title_font = ImageFont.truetype(FONT_PATH, 60)
            subtitle_font = ImageFont.truetype(FONT_PATH, 40)
            ticker_font = ImageFont.truetype(FONT_PATH, 30)
        except:
            print("⚠️  Hindi font not found, using default font")
            title_font = ImageFont.load_default()
            subtitle_font = ImageFont.load_default()
            ticker_font = ImageFont.load_default()
        
        # Use first video_with_text item for thumbnail
        main_headline = "Breaking News"
        sample_text = "Latest breaking news updates"
        for item in content:
            if item["segment_type"] == "video_with_text" and item["top_headline"]:
                main_headline = item["top_headline"]
                sample_text = item["text"][:100] + "..." if item["text"] else sample_text
                break
        
        if title_font:
            title_bbox = draw.textbbox((0, 0), main_headline, font=title_font)
            title_width = title_bbox[2] - title_bbox[0]
            title_x = (thumb_width - title_width) // 2
            title_y = 10
            draw.text((title_x + 3, title_y + 3), main_headline, fill='black', font=title_font)
            draw.text((title_x, title_y), main_headline, fill='white', font=title_font)
        
        breaking_news_text = "ताजा खबर"
        if subtitle_font:
            news_bbox = draw.textbbox((0, 0), breaking_news_text, font=subtitle_font)
            news_width = news_bbox[2] - news_bbox[0]
            news_x = 50
            news_y = header_height + 30
            padding = 15
            draw.rectangle([
                news_x - padding, 
                news_y - padding,
                news_x + news_width + padding,
                news_y + 50 + padding
            ], fill='#CC0000')
            draw.text((news_x + 2, news_y + 2), breaking_news_text, fill='black', font=subtitle_font)
            draw.text((news_x, news_y), breaking_news_text, fill='yellow', font=subtitle_font)
        
        if subtitle_font:
            wrapped_text = textwrap.fill(sample_text, width=50)
            text_y = 200
            for line in wrapped_text.split('\n')[:4]:
                text_bbox = draw.textbbox((0, 0), line, font=subtitle_font)
                text_width = text_bbox[2] - text_bbox[0]
                text_x = 50
                draw.text((text_x + 2, text_y + 2), line, fill='black', font=subtitle_font)
                draw.text((text_x, text_y), line, fill='white', font=subtitle_font)
                text_y += 45
        
        if logo_path and os.path.exists(logo_path):
            try:
                logo = Image.open(logo_path)
                logo_size = (120, 80)
                logo = logo.resize(logo_size, Image.Resampling.LANCZOS)
                logo_x = thumb_width - logo_size[0] - 30
                logo_y = 30
                if logo.mode == 'RGBA':
                    thumbnail.paste(logo, (logo_x, logo_y), logo)
                else:
                    thumbnail.paste(logo, (logo_x, logo_y))
            except Exception as e:
                print(f"⚠️  Could not add logo to thumbnail: {e}")
        
        # Use first available media for thumbnail
        media_path = None
        for item in content:
            if os.path.exists(item["media"]):
                media_path = item["media"]
                break
        
        if media_path:
            if media_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                temp_frame = f"{OUTPUT_DIR}/temp_frame.jpg"
                subprocess.run([
                    "ffmpeg", "-y", "-i", media_path, "-ss", "00:00:01", 
                    "-vframes", "1", "-q:v", "2", temp_frame
                ], capture_output=True)
                if os.path.exists(temp_frame):
                    media_img = Image.open(temp_frame)
                    media_size = (300, 200)
                    media_img = media_img.resize(media_size, Image.Resampling.LANCZOS)
                    media_x = thumb_width - media_size[0] - 50
                    media_y = thumb_height - media_size[1] - 50
                    draw.rectangle([
                        media_x - 5, media_y - 5,
                        media_x + media_size[0] + 5,
                        media_y + media_size[1] + 5
                    ], outline='white', width=3)
                    thumbnail.paste(media_img, (media_x, media_y))
                    os.remove(temp_frame)
            elif media_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                media_img = Image.open(media_path)
                media_size = (300, 200)
                media_img = media_img.resize(media_size, Image.Resampling.LANCZOS)
                media_x = thumb_width - media_size[0] - 50
                media_y = thumb_height - media_size[1] - 50
                draw.rectangle([
                    media_x - 5, media_y - 5,
                    media_x + media_size[0] + 5,
                    media_y + media_size[1] + 5
                ], outline='white', width=3)
                thumbnail.paste(media_img, (media_x, media_y))
        
        draw.rectangle([0, header_height, thumb_width, header_height + 3], fill='yellow')
        draw.rectangle([0, thumb_height - 3, thumb_width, thumb_height], fill='#CC0000')
        
        thumbnail_path = os.path.join(THUMBNAIL_DIR, f"bulletin_thumbnail_{uuid4().hex}.jpg")
        thumbnail.save(thumbnail_path, "JPEG", quality=95)
        
        print(f"✅ Bulletin thumbnail created: {thumbnail_path}")
        return thumbnail_path
    except Exception as e:
        print(f"❌ Thumbnail creation failed: {str(e)}")
        return None

def create_animated_text_filter(text, font_path, font_size, x, y, duration, line_height=85):
    """Create FFmpeg filter with perfect alignment and smooth page transitions"""
    all_words = text.split()
    
    if not all_words:
        return ""
    
    max_chars_per_line = 30
    max_lines_per_page = 7
    pages = create_enhanced_text_pages(all_words, max_chars_per_line, max_lines_per_page)
    
    print(f"📄 Text divided into {len(pages)} pages with perfect alignment")
    
    if len(pages) <= 1:
        return create_enhanced_single_page_animation(all_words, font_path, font_size, x, y, duration, line_height, max_chars_per_line)
    
    return create_smooth_multi_page_animation(pages, font_path, font_size, x, y, duration, line_height)

def create_enhanced_text_pages(words, max_chars_per_line, max_lines_per_page):
    """Create perfectly balanced pages with optimal text distribution that fits in the box"""
    pages = []
    current_page_lines = []
    current_line = ""
    
    i = 0
    while i < len(words):
        word = words[i]
        test_line = current_line + " " + word if current_line else word
        
        if len(test_line) <= max_chars_per_line:
            current_line = test_line
        else:
            if current_line:
                current_page_lines.append(current_line.strip())
                current_line = word
                if len(current_page_lines) >= max_lines_per_page:
                    pages.append(current_page_lines[:])
                    current_page_lines = []
            else:
                if len(word) > max_chars_per_line:
                    current_line = word[:max_chars_per_line-3] + "..."
                else:
                    current_line = word
        
        i += 1
    
    if current_line:
        current_page_lines.append(current_line.strip())
    
    if current_page_lines:
        pages.append(current_page_lines)
    
    validated_pages = []
    for page in pages:
        if len(page) > max_lines_per_page:
            validated_pages.append(page[:max_lines_per_page])
            remaining_lines = page[max_lines_per_page:]
            if remaining_lines:
                validated_pages.append(remaining_lines)
        else:
            validated_pages.append(page)
    
    print(f"📋 Created {len(validated_pages)} perfectly sized pages")
    return validated_pages

def create_enhanced_single_page_animation(words, font_path, font_size, x, y, duration, line_height, max_chars_per_line):
    """Enhanced single page animation with perfect word-by-word reveal that stays in box"""
    lines = create_optimal_text_lines(words, max_chars_per_line)
    if len(lines) > 7:
        lines = redistribute_text_to_fit(lines, max_chars_per_line, 7)
    
    print(f"📝 Single page: {len(lines)} lines, fits perfectly in box")
    
    animation_start_delay = 1.2
    animation_duration = duration - 2.5
    total_words = len(words)
    time_per_word = max(0.3, animation_duration / total_words)
    
    filters = []
    word_index = 0
    
    for line_num, line_text in enumerate(lines):
        line_words = line_text.split()
        line_y = y + (line_num * line_height)
        line_start_x = x + 25
        
        for word_pos, word in enumerate(line_words):
            word_start_time = animation_start_delay + (word_index * time_per_word)
            safe_word = escape_text_for_ffmpeg(word)
            word_x = line_start_x
            if word_pos > 0:
                prev_words = line_words[:word_pos]
                for prev_word in prev_words:
                    word_x += calculate_precise_word_width(prev_word, font_size) + 22
            
            word_filter = (
                f"drawtext=fontfile={font_path}:text='{safe_word}':"
                f"fontcolor=white:fontsize={font_size}:"
                f"x={int(word_x)}:y={int(line_y)}:"
                f"shadowcolor=black@0.95:shadowx=4:shadowy=4:"
                f"alpha='if(lt(t,{word_start_time:.2f}),0,if(lt(t,{word_start_time + 0.4:.2f}),(t-{word_start_time:.2f})/0.4,1))':"
                f"enable='gte(t,{word_start_time:.2f})'"
            )
            filters.append(word_filter)
            word_index += 1
    
    return ",".join(filters)

def create_smooth_multi_page_animation(pages, font_path, font_size, x, y, duration, line_height):
    """Create ultra-smooth page transition animations with perfect fade effects"""
    total_pages = len(pages)
    page_display_time = (duration - 2.0) / total_pages
    transition_duration = 1.0
    
    print(f"📺 Multi-page animation: {total_pages} pages, {page_display_time:.1f}s per page")
    
    filters = []
    
    for page_num, page_lines in enumerate(pages):
        page_start_time = 1.0 + (page_num * page_display_time)
        page_end_time = page_start_time + page_display_time - (transition_duration * 0.5)
        fade_out_start = page_end_time - 0.8
        
        total_page_words = sum(len(line.split()) for line in page_lines)
        word_animation_duration = min(page_display_time * 0.65, total_page_words * 0.25)
        time_per_word = word_animation_duration / total_page_words if total_page_words > 0 else 0.1
        
        word_index = 0
        
        for line_num, line_text in enumerate(page_lines):
            line_words = line_text.split()
            line_y = y + (line_num * line_height)
            line_start_x = x + 25
            
            for word_pos, word in enumerate(line_words):
                word_start_time = page_start_time + 0.4 + (word_index * time_per_word)
                safe_word = escape_text_for_ffmpeg(word)
                word_x = line_start_x
                if word_pos > 0:
                    prev_words = line_words[:word_pos]
                    for prev_word in prev_words:
                        word_x += calculate_precise_word_width(prev_word, font_size) + 22
                
                if word_start_time < fade_out_start:
                    if page_num < total_pages - 1:
                        alpha_expression = (
                            f"if(lt(t,{word_start_time:.2f}),0,"
                            f"if(lt(t,{word_start_time + 0.4:.2f}),(t-{word_start_time:.2f})/0.4,"
                            f"if(lt(t,{fade_out_start:.2f}),1,"
                            f"if(lt(t,{page_end_time:.2f}),1-((t-{fade_out_start:.2f})/{0.8:.2f}),0))))"
                        )
                    else:
                        alpha_expression = (
                            f"if(lt(t,{word_start_time:.2f}),0,"
                            f"if(lt(t,{word_start_time + 0.4:.2f}),(t-{word_start_time:.2f})/0.4,1))"
                        )
                    
                    word_filter = (
                        f"drawtext=fontfile={font_path}:text='{safe_word}':"
                        f"fontcolor=white:fontsize={font_size}:"
                        f"x={int(word_x)}:y={int(line_y)}:"
                        f"shadowcolor=black@0.95:shadowx=4:shadowy=4:"
                        f"alpha='{alpha_expression}':"
                        f"enable='between(t,{word_start_time:.2f},{page_end_time + 1.0:.2f})'"
                    )
                    filters.append(word_filter)
                word_index += 1
    
    return ",".join(filters)

def create_optimal_text_lines(words, max_chars_per_line):
    """Create optimally balanced text lines that fit perfectly in the box"""
    lines = []
    current_line = ""
    
    i = 0
    while i < len(words):
        word = words[i]
        test_line = current_line + " " + word if current_line else word
        
        if len(test_line) <= max_chars_per_line:
            current_line = test_line
        else:
            if current_line and len(current_line) >= max_chars_per_line * 0.5:
                lines.append(current_line.strip())
                current_line = word
            else:
                if len(test_line) <= max_chars_per_line + 2:
                    current_line = test_line
                else:
                    if current_line:
                        lines.append(current_line.strip())
                    current_line = word
        
        i += 1
    
    if current_line:
        lines.append(current_line.strip())
    
    return lines

def redistribute_text_to_fit(lines, max_chars_per_line, max_lines):
    """Redistribute text to fit exactly within the box constraints"""
    if len(lines) <= max_lines:
        return lines
    
    all_text = " ".join(lines)
    words = all_text.split()
    target_chars_per_line = min(max_chars_per_line + 5, 38)
    
    new_lines = []
    current_line = ""
    
    for word in words:
        test_line = current_line + " " + word if current_line else word
        
        if len(test_line) <= target_chars_per_line:
            current_line = test_line
        else:
            if current_line:
                new_lines.append(current_line.strip())
                current_line = word
            else:
                current_line = word
            
            if len(new_lines) >= max_lines - 1:
                remaining_words = words[words.index(word):]
                current_line = " ".join(remaining_words)
                break
    
    if current_line:
        new_lines.append(current_line.strip())
    
    return new_lines[:max_lines]

def calculate_precise_word_width(word, font_size):
    """More precise word width calculation for perfect alignment"""
    if not word:
        return 0
    
    hindi_chars = sum(1 for char in word if 0x0900 <= ord(char) <= 0x097F)
    english_chars = sum(1 for char in word if char.isalpha() and ord(char) < 256)
    digit_chars = sum(1 for char in word if char.isdigit())
    symbol_chars = len(word) - hindi_chars - english_chars - digit_chars
    
    width = (
        hindi_chars * (font_size * 0.78) +
        english_chars * (font_size * 0.55) +
        digit_chars * (font_size * 0.50) +
        symbol_chars * (font_size * 0.42)
    )
    
    return max(width, font_size * 0.35)

def escape_text_for_ffmpeg(text):
    """Enhanced text escaping for FFmpeg with better special character handling"""
    if not text:
        return ""
    
    escaped = text
    escaped = escaped.replace("'", "").replace('"', "").replace("`", "")
    escaped = escaped.replace("\\", "\\\\")
    escaped = escaped.replace(":", "\\:").replace("[", "\\[").replace("]", "\\]")
    escaped = escaped.replace("(", "\\(").replace(")", "\\)")
    escaped = escaped.replace(",", "\\,").replace(";", "\\;").replace("=", "\\=")
    escaped = escaped.replace("|", "\\|")
    escaped = escaped.replace("₹", "Rs.").replace("@", "\\@").replace("%", "\\%")
    escaped = escaped.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    while "  " in escaped:
        escaped = escaped.replace("  ", " ")
    
    return escaped.strip()

def generate_tts(text, out_path, language_code, language_name):
    """Generate TTS audio using Google Cloud Text-to-Speech with enhanced quality"""
    print(f"🎙️  Generating TTS for: {text[:50]}...")
    
    try:
        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code, 
            name=language_name,
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            speaking_rate=0.85,
            pitch=-2.0,
            volume_gain_db=6.0,
            sample_rate_hertz=44100
        )
        response = client.synthesize_speech(
            input=synthesis_input, 
            voice=voice, 
            audio_config=audio_config
        )
        
        if not response.audio_content:
            raise Exception("No audio content received from Google TTS")
        
        temp_wav = out_path.replace('.wav', '_raw.wav')
        with open(temp_wav, "wb") as out:
            out.write(response.audio_content)
        
        audio = AudioSegment.from_wav(temp_wav)
        audio = audio.set_frame_rate(44100).set_channels(2).set_sample_width(2)
        audio = audio.fade_in(50).fade_out(50)
        audio = audio.normalize()
        compressed_audio = audio.compress_dynamic_range(threshold=-20.0, ratio=4.0, attack=5.0, release=50.0)
        final_audio = compressed_audio + 2
        final_audio.export(
            out_path, 
            format="wav",
            parameters=["-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", "-f", "wav"]
        )
        
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
        
        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            raise Exception(f"Generated audio file is empty: {out_path}")
        
        test_audio = AudioSegment.from_wav(out_path)
        duration_ms = len(test_audio)
        
        print(f"✅ Audio verification passed: Duration {duration_ms/1000:.2f}s, Sample Rate {test_audio.frame_rate}Hz")
        return out_path
    except Exception as e:
        print(f"❌ TTS Generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"TTS Generation failed: {str(e)}")

def get_audio_duration(audio_path):
    """Get duration of audio file"""
    if audio_path.endswith('.mp3'):
        return MP3(audio_path).info.length
    else:
        audio = AudioSegment.from_wav(audio_path)
        return len(audio) / 1000.0

def generate_full_bulletin_audio(content, language_code, language_name, output_audio_path):
    """Generate full audio for the entire bulletin by concatenating all TTS segments"""
    print("🎵 Generating full bulletin audio...")
    
    try:
        all_audio_segments = []
        temp_audio_files = []
        
        # Generate TTS for each text segment
        for idx, item in enumerate(content):
            if item["segment_type"] == "video_with_text" and item["text"]:
                temp_audio_path = f"{OUTPUT_DIR}/temp_audio_{idx}.wav"
                temp_audio_files.append(temp_audio_path)
                
                # Generate TTS for this segment
                generate_tts(item["text"], temp_audio_path, language_code, language_name)
                
                # Load the audio segment
                audio_segment = AudioSegment.from_wav(temp_audio_path)
                all_audio_segments.append(audio_segment)
                
                # Add a small pause between segments (0.5 seconds)
                pause = AudioSegment.silent(duration=500)
                all_audio_segments.append(pause)
        
        if not all_audio_segments:
            print("⚠️  No text segments found for audio generation")
            return None
        
        # Concatenate all audio segments
        full_audio = sum(all_audio_segments)
        
        # Apply final processing
        full_audio = full_audio.normalize()
        full_audio = full_audio.fade_in(100).fade_out(100)
        
        # Export the full audio
        full_audio.export(
            output_audio_path,
            format="wav",
            parameters=["-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", "-f", "wav"]
        )
        
        # Cleanup temporary files
        for temp_file in temp_audio_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        duration_seconds = len(full_audio) / 1000.0
        print(f"✅ Full bulletin audio generated: {output_audio_path} (Duration: {duration_seconds:.2f}s)")
        
        return output_audio_path, duration_seconds
    
    except Exception as e:
        print(f"❌ Full audio generation failed: {str(e)}")
        # Cleanup temporary files on error
        for temp_file in temp_audio_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
        raise HTTPException(status_code=500, detail=f"Full audio generation failed: {str(e)}")

def create_main_layout(bg, audio, text, media_clip, out_path, top_headline, duration):
    """Create layout with improved text positioning and perfect box alignment"""
    print(f"🎨 Creating layout with logo and perfectly aligned animated text...")
    
    VIDEO_WIDTH = 1920
    VIDEO_HEIGHT = 1080
    headline_height = 100
    headline_y = 20
    logo_width = 120
    logo_height = 80
    logo_x = VIDEO_WIDTH - logo_width - 30
    logo_y = 30
    text_box_width = 960
    text_box_height = 600
    text_box_x = 0
    text_box_y = 150
    media_window_width = 800   
    media_window_height = 600  
    media_window_x = 1060  
    media_window_y = 150   
    text_x = text_box_x + 20
    text_y = text_box_y + 40
    font_size = 48
    line_height = 75
    news_label_y = 850
    news_label_height = 80
    
    safe_headline = escape_text_for_ffmpeg(top_headline)
    animated_text_filter = create_animated_text_filter(
        text, FONT_PATH, font_size, text_x, text_y, duration, line_height
    )
    
    filter_string = (
        f"[0:v]scale={VIDEO_WIDTH}:{VIDEO_HEIGHT},format=yuv420p[bg_scaled];"
        f"[bg_scaled]drawbox=x=0:y={headline_y}:w={VIDEO_WIDTH}:h={headline_height}:color=0xCC0000@1.0:t=fill[headline_bg];"
        f"[headline_bg]drawtext=fontfile={FONT_PATH}:text='{safe_headline}':fontcolor=white:fontsize=42:x=80:y={headline_y + 25}:shadowcolor=black@0.8:shadowx=3:shadowy=3:enable='gte(t,0.5)'[headline_applied];"
        f"[headline_applied]drawbox=x={text_box_x}:y={text_box_y}:w={text_box_width}:h={text_box_height}:color=0x003399@0.95:t=fill[text_bg];"
        f"[text_bg]{animated_text_filter}[text_applied];"
        f"[text_applied]drawbox=x={media_window_x - 10}:y={media_window_y - 10}:w={media_window_width + 20}:h={media_window_height + 20}:color=white@1.0:t=fill[media_border];"
        f"[media_border][1:v]overlay={media_window_x}:{media_window_y}:enable='gte(t,1.0)'[media_applied];"
        f"[3:v]scale={logo_width}:{logo_height}[logo_scaled];"
        f"[media_applied][logo_scaled]overlay={logo_x}:{logo_y}:enable='gte(t,0.8)'[logo_applied];"
        f"[logo_applied]drawbox=x=0:y={news_label_y}:w=450:h={news_label_height}:color=0xCC0000@1.0:t=fill[news_label_bg];"
        f"[news_label_bg]drawtext=fontfile={FONT_PATH}:text='ताजा खबर':fontcolor=yellow:fontsize=48:x=25:y={news_label_y + 18}:shadowcolor=black@0.8:shadowx=3:shadowy=3:enable='gte(t,2.0)'[final]"
    )
    
    cmd = [
        "ffmpeg", "-y",
        "-i", bg,
        "-i", media_clip,
        "-i", audio,
        "-i", LOGO_IMAGE,
        "-filter_complex", filter_string,
        "-map", "[final]",
        "-map", "2:a",
        "-c:v", "libx264",
        "-c:a", "aac",
        "-b:a", "128k",
        "-ar", "22050",
        "-preset", "medium",
        "-crf", "21",
        "-shortest", out_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Layout creation error: {result.stderr}")
        raise Exception("Layout creation failed")
    
    print(f"✅ Layout with perfectly aligned text generated: {out_path}")
    return out_path

def create_fullscreen_video(media_path, duration, out_path):
    """Create a full-screen video without overlays"""
    print(f"🎬 Creating full-screen video: {media_path}")
    
    VIDEO_WIDTH = 1920
    VIDEO_HEIGHT = 1080
    
    ext = os.path.splitext(media_path)[1].lower()
    
    if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        cmd = [
            "ffmpeg", "-y",
            "-i", media_path,
            "-vf", f"scale={VIDEO_WIDTH}:{VIDEO_HEIGHT}:force_original_aspect_ratio=decrease,pad={VIDEO_WIDTH}:{VIDEO_HEIGHT}:(ow-iw)/2:(oh-ih)/2,format=yuv420p",
            "-c:v", "libx264",
            "-t", str(duration),
            "-an",
            "-preset", "medium",
            "-crf", "23",
            out_path
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-i", media_path,
            "-vf", f"scale={VIDEO_WIDTH}:{VIDEO_HEIGHT}:force_original_aspect_ratio=decrease,pad={VIDEO_WIDTH}:{VIDEO_HEIGHT}:(ow-iw)/2:(oh-ih)/2,format=yuv420p,loop={(int(duration/video_duration(media_path))+1) if video_duration(media_path) < duration else 0}",
            "-c:v", "libx264",
            "-t", str(duration),
            "-an",
            "-preset", "medium",
            "-crf", "23",
            out_path
        ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Full-screen video creation error: {result.stderr}")
        raise Exception("Full-screen video creation failed")
    
    print(f"✅ Full-screen video created: {out_path}")
    return out_path

def video_duration(media_path):
    """Get duration of a video file"""
    result = subprocess.run([
        "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", media_path
    ], capture_output=True, text=True)
    return float(result.stdout.strip())

def prepare_media_clip(media, duration, output):
    """Prepare media for right side positioning - half screen coverage"""
    print(f"🎬 Preparing media clip: {media}")
    
    ext = os.path.splitext(media)[1].lower()
    
    target_width = 800
    target_height = 600
    
    if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        clip = (ImageClip(media)
                .set_duration(duration)
                .resize((target_width, target_height))
                .set_position("center"))
    else:
        video = VideoFileClip(media)
        loops = int(duration / video.duration) + 1
        if loops > 1:
            looped_video = concatenate_videoclips([video] * loops)
            final_video = looped_video.subclip(0, duration)
        else:
            final_video = video.subclip(0, min(duration, video.duration))
        clip = final_video.resize((target_width, target_height)).set_position("center")
    
    clip.write_videofile(output, fps=25, codec="libx264", audio=False, verbose=False, logger=None)
    clip.close()
    print(f"✅ Media clip prepared: {output}")

def overlay_frame(video_with_layout, frame, out_path):
    """Overlay the frame on top of everything with proper scaling and animation"""
    print(f"🖼️  Adding animated frame overlay...")
    
    cmd = [
        "ffmpeg", "-y",
        "-i", video_with_layout,
        "-i", frame,
        "-filter_complex",
        "[1:v]scale=1920:1080[frame_scaled];[0:v][frame_scaled]overlay=0:0:enable='gte(t,0.2)'[final]",
        "-map", "[final]",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-c:a", "copy",
        "-preset", "medium",
        "-crf", "23",
        "-shortest", out_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Frame overlay error: {result.stderr}")
        raise Exception(f"Frame overlay failed: {result.stderr}")
    
    print(f"✅ Animated frame overlay completed: {out_path}")

def add_ticker(input_video, text, output_video):
    """Add professional scrolling ticker at bottom with animation effects"""
    print(f"📺 Adding animated ticker...")
    
    safe_text = escape_text_for_ffmpeg(text)
    repeated_text = (safe_text + "     ") * 3
    
    cmd = [
        "ffmpeg", "-y",
        "-i", input_video,
        "-vf",
        f"drawbox=y=960:color=red@1.0:width=1920:height=120:t=fill:enable='gte(t,3.0)',"
        f"drawbox=y=955:color=white@0.8:width=1920:height=5:t=fill:enable='gte(t,3.0)',"
        f"drawtext=fontfile='{FONT_PATH}':text='{repeated_text}':"
        f"fontsize=52:fontcolor=white:x=1920-mod(t*150\\,1920+tw):y=1005:"
        f"shadowcolor=black@0.9:shadowx=3:shadowy=3:enable='gte(t,3.5)'",
        "-codec:a", "copy",
        "-preset", "medium",
        output_video
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Ticker error: {result.stderr}")
        raise Exception(f"Ticker failed: {result.stderr}")
    
    print(f"✅ Animated ticker added: {output_video}")

def generate_segment(item, index, language_code, language_name, frame_path, bg_path):
    """Generate a single news segment based on segment type"""
    print(f"\n📝 Processing segment {index + 1}: {item['text'][:30] if item['segment_type'] == 'video_with_text' else '(No text)'}...")
    
    final_segment = f"{OUTPUT_DIR}/final_{index}.mp4"
    
    if item["segment_type"] == "video_without_text":
        duration = video_duration(item["media"])
        create_fullscreen_video(item["media"], duration, final_segment)
    else:
        audio_path = f"{OUTPUT_DIR}/audio_{index}.wav"
        media_clip = f"{OUTPUT_DIR}/media_{index}.mp4"
        layout_video = f"{OUTPUT_DIR}/layout_{index}.mp4"
        
        generate_tts(item["text"], audio_path, language_code, language_name)
        duration = get_audio_duration(audio_path)
        print(f"⏱️  Segment duration: {duration:.2f} seconds")
        
        prepare_media_clip(item["media"], duration, media_clip)
        create_main_layout(bg_path, audio_path, item["text"], media_clip, layout_video, item["top_headline"], duration)
        overlay_frame(layout_video, frame_path, final_segment)
    
    print(f"✅ Segment {index + 1} completed: {final_segment}")
    return final_segment

@app.post("/generate-bulletin")
def generate_bulletin(data: ContentInput):
    """Generate news bulletin from JSON API input without intro/outro videos"""
    request_id = uuid4().hex
    CLIPS_DIR = os.path.join(OUTPUT_DIR, request_id)
    os.makedirs(CLIPS_DIR, exist_ok=True)
    
    try:
        # Define default static assets
        DEFAULT_FRAME = "assets/frame.png"
        DEFAULT_BACKGROUND = "assets/bg/earth_bg.mp4"
        
        # Validate static assets
        required_static_assets = [FONT_PATH]
        missing_static_assets = [asset for asset in required_static_assets if not os.path.exists(asset)]
        if missing_static_assets:
            raise HTTPException(status_code=400, detail=f"Missing required static assets: {missing_static_assets}")
        
        # Test Google Cloud TTS configuration
        try:
            test_client = texttospeech.TextToSpeechClient()
            print("✅ Google Cloud TTS client initialized successfully")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Google Cloud TTS setup error: {str(e)}")
        
        # Download logo if provided
        global LOGO_IMAGE
        LOGO_IMAGE = os.path.join(CLIPS_DIR, f"logo_{uuid4().hex}.png") if data.logo_url else None
        if data.logo_url:
            LOGO_IMAGE = download_file(data.logo_url, CLIPS_DIR, "logo")
            logo_img = Image.open(LOGO_IMAGE).convert("RGBA")
            logo_img.thumbnail((120, 80), Image.Resampling.LANCZOS)
            logo_img.save(LOGO_IMAGE)
        
        # Download frame and background
        frame_path = download_file(data.frame_url, CLIPS_DIR, "frame") if data.frame_url else DEFAULT_FRAME
        bg_path = download_file(data.background_url, CLIPS_DIR, "background") if data.background_url else DEFAULT_BACKGROUND
        
        # Validate downloaded or default assets
        if not os.path.exists(frame_path) or os.path.getsize(frame_path) == 0:
            raise HTTPException(status_code=400, detail=f"Frame file is missing or empty: {frame_path}")
        if not os.path.exists(bg_path) or os.path.getsize(bg_path) == 0:
            raise HTTPException(status_code=400, detail=f"Background file is missing or empty: {bg_path}")
        
        # Download media files and update content
        content = []
        for item in data.content:
            media_path = download_file(item.media_url, CLIPS_DIR, f"media_{len(content)}")
            content.append({
                "segment_type": item.segment_type,
                "media": media_path,
                "text": item.text,
                "top_headline": item.top_headline
            })
        
        # Generate thumbnail
        thumbnail_path = create_bulletin_thumbnail(content, LOGO_IMAGE)
        if thumbnail_path:
            new_thumbnail_path = os.path.join(THUMBNAIL_DIR, os.path.basename(thumbnail_path))
            shutil.move(thumbnail_path, new_thumbnail_path)
            thumbnail_path = new_thumbnail_path
        
        # Generate full bulletin audio
        full_audio_path = os.path.join(CLIPS_DIR, f"full_bulletin_audio_{request_id}.wav")
        audio_result = generate_full_bulletin_audio(content, data.language_code, data.language_name, full_audio_path)
        
        if audio_result:
            final_audio_path, total_audio_duration = audio_result
            # Move to final location
            audio_output_path = os.path.join(OUTPUT_DIR, f"bulletin_audio_{request_id}.wav")
            shutil.move(final_audio_path, audio_output_path)
        else:
            audio_output_path = None
            total_audio_duration = 0
        
        # Generate all segments
        all_segments = []
        for idx, item in enumerate(content):
            print(f"\n{'='*60}")
            print(f"🎯 Processing Segment {idx + 1}/{len(content)}")
            print(f"📝 Type: {item['segment_type']}")
            seg = generate_segment(
                item,
                idx,
                data.language_code,
                data.language_name,
                frame_path,
                bg_path
            )
            all_segments.append(seg)
            if not os.path.exists(seg) or os.path.getsize(seg) == 0:
                raise Exception(f"Generated segment is empty: {seg}")
        
        # Merge segments
        merged_segments = os.path.join(CLIPS_DIR, f"full_bulletin_{request_id}.mp4")
        concat_list = os.path.join(CLIPS_DIR, "concat_list.txt")
        with open(concat_list, "w") as f:
            for seg in all_segments:
                f.write(f"file '{os.path.abspath(seg)}'\n")
        
        result = subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", concat_list, "-c", "copy", merged_segments
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Merge error: {result.stderr}")
        
        # Add ticker only if there are video_with_text segments
        has_text_segments = any(item["segment_type"] == "video_with_text" for item in content)
        if has_text_segments:
            final_output = os.path.join(VIDEO_DIR, f"final_bulletin_{request_id}.mp4")
            add_ticker(merged_segments, data.ticker, final_output)
        else:
            final_output = os.path.join(VIDEO_DIR, f"final_bulletin_{request_id}.mp4")
            shutil.move(merged_segments, final_output)
        
        # Get file info
        file_size = os.path.getsize(final_output) / (1024 * 1024)
        duration = float(subprocess.run([
            "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", final_output
        ], capture_output=True, text=True).stdout.strip())
        
        audio_size = os.path.getsize(audio_output_path) / (1024 * 1024) if audio_output_path and os.path.exists(audio_output_path) else 0
        
        response = {
            "message": "✅ Bulletin created successfully (without intro/outro)",
            "video_path": os.path.basename(final_output),
            "audio_path": os.path.basename(audio_output_path) if audio_output_path else None,
            "thumbnail_path": os.path.basename(thumbnail_path) if thumbnail_path else None,
            "file_size_mb": round(file_size, 2),
            "audio_size_mb": round(audio_size, 2),
            "duration_seconds": round(duration, 1),
            "audio_duration_seconds": round(total_audio_duration, 1) if audio_output_path else 0
        }
        
        return JSONResponse(response)
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    
    finally:
        print("\n🧹 Cleaning up intermediate files...")
        cleanup_files = [
            f"{CLIPS_DIR}/audio_*.wav",
            f"{CLIPS_DIR}/media_*.mp4",
            f"{CLIPS_DIR}/layout_*.mp4",
            f"{CLIPS_DIR}/final_*.mp4",
            f"{CLIPS_DIR}/full_bulletin_*.mp4",
            f"{CLIPS_DIR}/concat_list.txt",
            f"{CLIPS_DIR}/logo_*.png",
            f"{CLIPS_DIR}/frame_*.png",
            f"{CLIPS_DIR}/background_*.mp4"
        ]
        for pattern in cleanup_files:
            for file in glob.glob(pattern):
                try:
                    os.remove(file)
                    print(f"🗑️  Removed: {os.path.basename(file)}")
                except:
                    pass
        
        try:
            shutil.rmtree(CLIPS_DIR)
            print(f"🧹 Removed directory: {CLIPS_DIR}")
        except:
            pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)