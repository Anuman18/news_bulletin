import os
import subprocess
import textwrap
import requests
import time
import re
import unicodedata
import uuid
import shutil
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from moviepy.editor import VideoFileClip, ImageClip, concatenate_videoclips
from mutagen.mp3 import MP3
from google.cloud import texttospeech

app = FastAPI()

# Setup paths
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key.json"
OUTPUT_DIR = "output"
VIDEO_BULLETIN_DIR = "video_bulletin"
THUMBNAIL_DIR = "thumbnail"
FONT_PATH = "assets/NotoSans-Regular.ttf"
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

# === Enhanced Hindi Text Processing Functions ===
def fix_hindi_text(text):
    """
    Enhanced Hindi text processing with proper Unicode normalization
    and comprehensive matra corrections
    """
    if not text:
        return text
    
    # Step 1: Normalize Unicode to NFC (Canonical Decomposition followed by Canonical Composition)
    text = unicodedata.normalize('NFC', text)
    
    # Step 2: Fix common matra placement issues
    matra_patterns = [
        # Fix misplaced 'i' matra (ि) - should come before consonant
        (r'([क-ह])ि', r'ि\1'),
        
        # Fix double matras or incorrect combinations
        (r'([क-ह])ी([ं])', r'\1ीं'),  # Fix ी + ं combinations
        (r'([क-ह])े([ं])', r'\1ें'),  # Fix े + ं combinations
        (r'([क-ह])ो([ं])', r'\1ों'),  # Fix ो + ं combinations
        
        # Fix common word-level corrections
        (r'मे\b', 'में'),     # में not मे
        (r'हे\b', 'है'),     # है not हे
        (r'गै\b', 'गए'),     # गए not गै
        (r'कै\b', 'कए'),     # कए not कै
        (r'जै\b', 'जए'),     # जए not जै
        (r'तै\b', 'तए'),     # तए not तै
        (r'दै\b', 'दए'),     # दए not दै
        (r'नै\b', 'नए'),     # नए not नै
        (r'बै\b', 'बए'),     # बए not बै
        (r'रै\b', 'रए'),     # रए not रै
        (r'सै\b', 'सए'),     # सए not सै
        (r'लै\b', 'लए'),     # लए not लै
        (r'पै\b', 'पए'),     # पए not पै
    ]
    
    # Apply matra pattern fixes
    for pattern, replacement in matra_patterns:
        text = re.sub(pattern, replacement, text)
    
    # Step 3: Fix common vocabulary and spelling mistakes
    vocabulary_fixes = {
        # Common mistakes - most important fixes
        'मै': 'मैं',        # Very common mistake
        'मे': 'में',        # में not मे
        'हे': 'है',        # है not हे
        'गै': 'गए',        # गए not गै
        'कै': 'कए',        # कए not कै
        'जै': 'जए',        # जए not जै
        'तै': 'तए',        # तए not तै
        'दै': 'दए',        # दए not दै
        'नै': 'नए',        # नए not नै
        'बै': 'बए',        # बए not बै
        'रै': 'रए',        # रए not रै
        'सै': 'सए',        # सए not सै
        r'\bलै\b': 'लए',        # लए not लै
        'पै': 'पए',        # पए not पै
        
        # Common words
        'बारे': 'बारे',
        'साथ': 'साथ',
        'बाद': 'बाद',
        'पहले': 'पहले',
        'अभी': 'अभी',
        'बहुत': 'बहुत',
        'सभी': 'सभी',
        'कोई': 'कोई',
        'कुछ': 'कुछ',
        'जब': 'जब',
        'तब': 'तब',
        'कहाँ': 'कहाँ',
        'कैसे': 'कैसे',
        'क्यों': 'क्यों',
        'क्या': 'क्या',
        
        # English to Hindi fixes
        'न्यूज़': 'समाचार',
        'न्यूज': 'समाचार',
        'रिपोर्ट': 'रिपोर्ट',
        'अपडेट': 'अद्यतन',
        'ब्रेकिंग': 'तत्काल',
        'लाइव': 'प्रत्यक्ष',
        'चैनल': 'चैनल',
        'वीडियो': 'वीडियो',
        'टेक्स्ट': 'टेक्स्ट',
        
        # Government terms
        'गवर्नमेंट': 'सरकार',
        'मिनिस्टर': 'मंत्री',
        'प्राइम मिनिस्टर': 'प्रधानमंत्री',
        'प्रेसिडेंट': 'राष्ट्रपति',
        'चीफ मिनिस्टर': 'मुख्यमंत्री',
        
        # Common greetings
        'हेलो': 'नमस्कार',
        'हाय': 'नमस्ते',
        'थैंक्स': 'धन्यवाद',
        'सॉरी': 'क्षमा करें',
        'प्लीज़': 'कृपया',
        'ओके': 'ठीक है',
        'यस': 'हाँ',
        'नो': 'नहीं',
    }
    
    # Apply vocabulary fixes with word boundaries
    for wrong, correct in vocabulary_fixes.items():
        pattern = r'\b' + re.escape(wrong) + r'\b'
        text = re.sub(pattern, correct, text)
    
    # Step 4: Add respectful prefixes for dignitary references
    dignitary_words = ['मंत्री', 'प्रधानमंत्री', 'राष्ट्रपति', 'मुख्यमंत्री', 'राज्यपाल']
    for word in dignitary_words:
        if word in text and not any(prefix in text for prefix in ['माननीय', 'श्री', 'श्रीमती', 'डॉ']):
            text = re.sub(r'\b' + word + r'\b', f'माननीय {word}', text, count=1)
    
    # Step 5: Ensure proper sentence endings
    if text.strip() and not text.strip().endswith(('।', '.', '!', '?')):
        text = text.strip() + '।'
    
    # Step 6: Clean up extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def calculate_hindi_width(text):
    """
    Calculate accurate display width for Hindi text considering matras and conjuncts
    """
    if not text:
        return 0
    
    width = 0
    i = 0
    
    while i < len(text):
        char = text[i]
        category = unicodedata.category(char)
        
        # Get Unicode name for better classification
        try:
            char_name = unicodedata.name(char, '')
        except:
            char_name = ''
        
        if category == 'Mn':  # Nonspacing mark (most matras)
            # These don't add to width as they combine with previous character
            pass
        elif category == 'Mc':  # Spacing combining mark
            # These add minimal width
            width += 0.3
        elif char == ' ':
            width += 1
        elif '\u0900' <= char <= '\u097F':  # Devanagari block
            if 'VOWEL SIGN' in char_name:
                # Vowel signs (matras) - some are spacing, some aren't
                if char in 'ािीुूृेৈোৌ':  # Common matras
                    width += 0.5 if char in 'ािुূ' else 0.7
                else:
                    width += 0.5
            else:
                # Regular Devanagari characters
                width += 1
        elif category.startswith('L'):  # Letter
            width += 1
        else:
            # Punctuation, numbers, etc.
            width += 0.5 if char in '।,;:' else 1
        
        i += 1
    
    return int(width + 0.5)  # Round up

def wrap_hindi_text(text, max_chars_per_line=25):
    """
    Intelligent Hindi text wrapping with proper word boundary handling
    """
    # First, fix the text
    fixed_text = fix_hindi_text(text)
    
    # Split into words while preserving punctuation
    words = re.findall(r'\S+', fixed_text)
    
    lines = []
    current_line = ""
    
    for word in words:
        word_width = calculate_hindi_width(word)
        current_width = calculate_hindi_width(current_line)
        
        # Check if adding this word would exceed the limit
        space_needed = 1 if current_line else 0
        if current_line and (current_width + space_needed + word_width) > max_chars_per_line:
            # Start new line
            lines.append(current_line.strip())
            current_line = word
        else:
            # Add to current line
            if current_line:
                current_line += " " + word
            else:
                current_line = word
    
    # Add the last line
    if current_line.strip():
        lines.append(current_line.strip())
    
    return '\n'.join(lines)

def split_hindi_text_into_pages(text, max_chars_per_page=150):
    """
    Split Hindi text into pages with intelligent sentence handling
    """
    # Fix the text first
    fixed_text = fix_hindi_text(text)
    
    # Calculate total display width
    total_width = calculate_hindi_width(fixed_text)
    
    # If it fits in one page, return as is
    if total_width <= max_chars_per_page:
        return [fixed_text]
    
    # Split into sentences using Hindi punctuation
    sentence_endings = r'([।\.!\?])'
    sentences = re.split(sentence_endings, fixed_text)
    
    # Recombine sentences with their punctuation
    complete_sentences = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            sentence = (sentences[i] + sentences[i + 1]).strip()
            if sentence:
                complete_sentences.append(sentence)
    
    # Handle case where last sentence doesn't have punctuation
    if len(sentences) % 2 == 1 and sentences[-1].strip():
        complete_sentences.append(sentences[-1].strip())
    
    # Group sentences into pages
    pages = []
    current_page = []
    current_page_width = 0
    
    for sentence in complete_sentences:
        sentence_width = calculate_hindi_width(sentence)
        
        # If adding this sentence would exceed the limit and we have content
        if current_page and (current_page_width + sentence_width + 1) > max_chars_per_page:
            # Finalize current page
            pages.append(' '.join(current_page))
            current_page = [sentence]
            current_page_width = sentence_width
        else:
            # Add sentence to current page
            current_page.append(sentence)
            current_page_width += sentence_width + (1 if len(current_page) > 1 else 0)
    
    # Add the last page if it has content
    if current_page:
        pages.append(' '.join(current_page))
    
    return pages

def validate_hindi_unicode(text):
    """
    Validate if Hindi text has proper Unicode encoding
    """
    try:
        # Try to encode and decode to check for issues
        encoded = text.encode('utf-8')
        decoded = encoded.decode('utf-8')
        
        # Check if it's properly normalized
        normalized = unicodedata.normalize('NFC', text)
        
        return {
            'is_valid_utf8': decoded == text,
            'is_normalized': normalized == text,
            'has_devanagari': any('\u0900' <= char <= '\u097F' for char in text),
            'original_length': len(text),
            'normalized_length': len(normalized),
            'display_width': calculate_hindi_width(text)
        }
    except Exception as e:
        return {
            'is_valid_utf8': False,
            'error': str(e)
        }

def clean_hindi_input_text(text):
    """
    Clean and prepare Hindi text input with comprehensive processing
    """
    if not text or not text.strip():
        return ""
    
    # Step 1: Basic cleanup
    text = text.strip()
    
    # Step 2: Remove extra whitespaces and normalize
    text = re.sub(r'\s+', ' ', text)
    
    # Step 3: Fix encoding issues if any
    try:
        # Handle common encoding problems
        text = text.encode('utf-8').decode('utf-8')
    except:
        pass
    
    # Step 4: Apply our comprehensive Hindi fixes
    text = fix_hindi_text(text)
    
    # Step 5: Final validation
    validation = validate_hindi_unicode(text)
    if not validation.get('is_valid_utf8', True):
        print(f"⚠️ Warning: Text may have encoding issues")
    
    return text

def process_content_item_text(text, max_chars_per_page=150):
    """
    Process a content item's text with improved Hindi handling
    """
    if not text:
        return []
    
    # Clean and fix the text
    cleaned_text = clean_hindi_input_text(text)
    
    # Split into pages with proper handling
    pages = split_hindi_text_into_pages(cleaned_text, max_chars_per_page)
    
    # Validate each page
    validated_pages = []
    for page in pages:
        validation = validate_hindi_unicode(page)
        if validation.get('is_valid_utf8', True):
            validated_pages.append(page)
        else:
            print(f"⚠️ Warning: Skipping invalid page: {page[:50]}...")
    
    return validated_pages

def debug_hindi_text(text):
    """
    Debug function to analyze Hindi text character by character
    """
    print(f"🔍 Debugging text: {text}")
    print(f"📏 Length: {len(text)} characters")
    print(f"📐 Display width: {calculate_hindi_width(text)}")
    
    print("\n📋 Character analysis:")
    for i, char in enumerate(text):
        try:
            name = unicodedata.name(char)
            category = unicodedata.category(char)
            codepoint = ord(char)
            print(f"  {i:2d}: '{char}' U+{codepoint:04X} {category} {name}")
        except:
            print(f"  {i:2d}: '{char}' U+{ord(char):04X} {unicodedata.category(char)} [Unknown]")
    
    print(f"\n✅ Fixed version: {fix_hindi_text(text)}")

# === Helper functions ===
def download_file(url, filename, temp_dir):
    """Download file to request-specific temp directory"""
    path = os.path.join(temp_dir, filename)
    try:
        print(f"🔄 Attempting to download: {url}")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        r = requests.get(url, headers=headers, timeout=30, stream=True, verify=False)
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
            
            print(f"✅ Successfully downloaded: {filename}")
            return path
        else:
            raise Exception(f"HTTP {r.status_code}: Failed to download from {url}")
            
    except requests.exceptions.ConnectionError as e:
        print(f"❌ Connection Error: Cannot reach {url}")
        raise Exception(f"Connection failed for {url}")
    
    except requests.exceptions.Timeout as e:
        print(f"⏰ Timeout Error: {url} took too long to respond")
        raise Exception(f"Download timeout for {url}")
    
    except Exception as e:
        print(f"💥 Unexpected error downloading {url}: {str(e)}")
        raise Exception(f"Download failed: {str(e)}")

def generate_tts(text, voice_name, lang_code, output_path):
    """Generate TTS with properly processed Hindi text"""
    # Clean and fix text before TTS - this is crucial
    cleaned_text = clean_hindi_input_text(text)
    
    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=cleaned_text)
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
    """Extend media with seamless looping"""
    ext = os.path.splitext(media_path)[1].lower()
    
    if ext in ['.jpg', '.jpeg', '.png']:
        # For images, create a video
        cmd = [
            "ffmpeg", "-y",
            "-loop", "1",
            "-i", media_path,
            "-t", str(duration),
            "-vf", "scale=1100:620,format=yuv420p",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-pix_fmt", "yuv420p",
            output_path
        ]
        subprocess.run(cmd, check=True)
    else:
        # For videos, use stream_loop for seamless looping
        cmd = [
            "ffmpeg", "-y",
            "-stream_loop", "-1",
            "-i", media_path,
            "-t", str(duration),
            "-vf", "scale=1100:620,fps=25,format=yuv420p",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-g", "25",  # Keyframe every second
            "-bf", "0",   # No B-frames
            output_path
        ]
        subprocess.run(cmd, check=True)

def cleanup_request_files(request_temp_dir):
    """Clean up files for a specific request"""
    try:
        if os.path.exists(request_temp_dir):
            shutil.rmtree(request_temp_dir)
            print(f"✅ Cleaned up temporary directory: {request_temp_dir}")
    except Exception as e:
        print(f"⚠️ Warning: Could not clean up temp directory {request_temp_dir}: {e}")

def create_text_background(bg_path, text, audio_path, output_path, temp_dir):
    """Create background video with text overlay and TTS audio - Updated with Hindi processing"""
    font_size = 46
    font_path = os.path.abspath(FONT_PATH)
    subtitle_file = os.path.join(temp_dir, "tts_text.txt")
    breaking_news_file = os.path.join(temp_dir, "breaking_news.txt")

    # Fixed box dimensions
    box_width = 650
    box_height = 460  
    box_y = 220
    header_y = box_y + 10
    text_y = box_y + 70
    
    # Write "BREAKING NEWS" text in Hindi
    with open(breaking_news_file, "w", encoding="utf-8") as f:
        f.write("तत्काल समाचार")  # Use Hindi instead of English
    
    # Wrap text properly with improved Hindi processing
    wrapped_text = wrap_hindi_text(text, max_chars_per_line=25)
    
    with open(subtitle_file, "w", encoding="utf-8") as f:
        f.write(wrapped_text)

    # Get audio duration
    audio_duration = get_audio_duration(audio_path)
    
    assert os.path.exists(bg_path), "Missing background"
    assert os.path.exists(audio_path), "Missing TTS audio"
    assert os.path.exists(font_path), "Missing font"
    assert os.path.exists(subtitle_file), "Subtitle file missing"

    # Create looped background video
    temp_bg_looped = os.path.join(temp_dir, "bg_looped_temp.mp4")
    loop_cmd = [
        "ffmpeg", "-y",
        "-stream_loop", "-1",
        "-i", bg_path,
        "-t", str(audio_duration),
        "-c:v", "libx264",
        temp_bg_looped
    ]
    subprocess.run(loop_cmd, check=True)

    # Create filter with boxes and text
    filter_str = (
        f"[0:v]drawbox=x=40:y={box_y}:w={box_width}:h={box_height}:color=black@0.7:t=fill,"
        f"drawbox=x=40:y={box_y}:w={box_width}:h=50:color=red@1.0:t=fill,"
        f"drawtext=fontfile={font_path}:textfile={breaking_news_file}:"
        f"fontcolor=white:fontsize=36:x=(40+{box_width}/2-text_w/2):y={header_y}:"
        f"shadowcolor=black:shadowx=2:shadowy=2,"
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
        "-map", "1:a",
        "-c:v", "libx264",
        "-c:a", "aac",
        output_path
    ]
    subprocess.run(cmd, check=True)

def create_continuous_media_loop(media_files, total_duration, output_path, temp_dir):
    """Create a truly continuous looped video using stream_loop"""
    if not media_files:
        return
    
    # First, create a single concatenated video from all media files
    temp_concat = os.path.join(temp_dir, "temp_single_concat.mp4")
    
    if len(media_files) == 1:
        # If only one file, just copy it
        normalize_cmd = [
            "ffmpeg", "-y",
            "-i", media_files[0],
            "-vf", "scale=1100:620,fps=25,format=yuv420p",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "23",
            "-g", "25",  # Set keyframe interval
            "-bf", "0",  # No B-frames for better seeking
            temp_concat
        ]
        subprocess.run(normalize_cmd, check=True)
    else:
        # Normalize all videos first
        normalized_files = []
        for i, media_path in enumerate(media_files):
            normalized_path = os.path.join(temp_dir, f"norm_{i}.mp4")
            normalize_cmd = [
                "ffmpeg", "-y",
                "-i", media_path,
                "-vf", "scale=1100:620,fps=25,format=yuv420p",
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", "23",
                "-g", "25",
                "-bf", "0",
                normalized_path
            ]
            subprocess.run(normalize_cmd, check=True)
            normalized_files.append(normalized_path)
        
        # Create concat filter
        inputs = []
        filter_parts = []
        for i, norm_file in enumerate(normalized_files):
            inputs.extend(["-i", norm_file])
            filter_parts.append(f"[{i}:v:0]")
        
        filter_complex = "".join(filter_parts) + f"concat=n={len(normalized_files)}:v=1:a=0[outv]"
        
        concat_cmd = [
            "ffmpeg", "-y"
        ] + inputs + [
            "-filter_complex", filter_complex,
            "-map", "[outv]",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "23",
            temp_concat
        ]
        subprocess.run(concat_cmd, check=True)
    
    # Now create the infinitely looped version using stream_loop
    loop_cmd = [
        "ffmpeg", "-y",
        "-stream_loop", "-1",  # Infinite loop
        "-i", temp_concat,
        "-t", str(total_duration * 2),  # Double the duration for safety
        "-c:v", "copy",  # Just copy, don't re-encode
        output_path
    ]
    subprocess.run(loop_cmd, check=True)
    
    single_duration = get_video_duration(temp_concat)
    print(f"✅ Created seamless looped media: base duration = {single_duration}s, output = {total_duration * 2}s")

def overlay_continuous_media(bg_text_video, continuous_media, frame_img, start_time, duration, output_path, temp_dir):
    """Overlay continuous media with better performance"""
    # Create a segment of the continuous media
    segment_path = os.path.join(temp_dir, f"segment_{int(start_time)}_{int(duration)}.mp4")
    
    # Extract exact segment needed with keyframe alignment
    segment_cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_time),
        "-i", continuous_media,
        "-t", str(duration),
        "-c:v", "copy",
        "-avoid_negative_ts", "make_zero",
        segment_path
    ]
    subprocess.run(segment_cmd, check=True)
    
    # Now overlay the segment
    overlay_cmd = [
        "ffmpeg", "-y",
        "-i", bg_text_video,
        "-i", segment_path,
        "-i", frame_img,
        "-filter_complex",
        "[1:v]scale=1100:620,setpts=PTS-STARTPTS[media];"
        "[0:v][media]overlay=760:220:shortest=1[tmp];"
        "[tmp][2:v]overlay=0:0[out]",
        "-map", "[out]",
        "-map", "0:a",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-c:a", "aac",
        output_path
    ]
    subprocess.run(overlay_cmd, check=True)

def create_smooth_looped_video(media_path, duration, output_path):
    """Create a smooth looped video using a different approach"""
    # Get the duration of the source video
    source_duration = get_video_duration(media_path)
    
    if source_duration >= duration:
        # If source is longer than needed, just trim it
        cmd = [
            "ffmpeg", "-y",
            "-i", media_path,
            "-t", str(duration),
            "-vf", "scale=1100:620,fps=25,format=yuv420p",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            output_path
        ]
    else:
        # Create a smooth loop by crossfading
        loops_needed = int(duration / source_duration) + 2
        
        # First, create multiple copies
        filter_complex = ""
        inputs = []
        
        for i in range(loops_needed):
            inputs.extend(["-i", media_path])
        
        # Build filter for smooth transitions
        filter_parts = []
        for i in range(loops_needed):
            filter_parts.append(f"[{i}:v]scale=1100:620,fps=25,format=yuv420p[v{i}];")
        
        # Concatenate with xfade for smooth transitions
        concat_filter = "".join(filter_parts)
        for i in range(loops_needed):
            concat_filter += f"[v{i}]"
        concat_filter += f"concat=n={loops_needed}:v=1:a=0[out]"
        
        cmd = [
            "ffmpeg", "-y"
        ] + inputs + [
            "-filter_complex", concat_filter,
            "-map", "[out]",
            "-t", str(duration),
            "-c:v", "libx264",
            "-preset", "ultrafast",
            output_path
        ]
    
    subprocess.run(cmd, check=True)

# Add this helper function for better video processing
def preprocess_video_for_looping(input_path, output_path):
    """Preprocess video to ensure smooth looping"""
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vf", "scale=1100:620,fps=25,format=yuv420p,setpts=PTS-STARTPTS",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "23",
        "-x264-params", "keyint=25:min-keyint=25:scenecut=0",  # Regular keyframes
        "-movflags", "+faststart",
        "-avoid_negative_ts", "make_zero",
        output_path
    ]
    subprocess.run(cmd, check=True)

def generate_thumbnail(video_path, thumbnail_path):
    """Generate high-quality thumbnail from video with better frame selection"""
    if not os.path.exists(video_path):
        raise Exception(f"Video file not found: {video_path}")

    try:
        # Method 1: Try to get a frame from 2 seconds (usually when content is stable)
        cmd = [
            "ffmpeg", "-y",
            "-ss", "00:00:02",
            "-i", video_path,
            "-vf", "select='eq(pict_type,I)',scale=1920:1080,unsharp=5:5:1.0:5:5:0.0",
            "-vframes", "1",
            "-q:v", "2",  # High quality JPEG
            thumbnail_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0 or not os.path.exists(thumbnail_path):
            # Method 2: If that fails, try the middle of the video
            duration = get_video_duration(video_path)
            middle_time = duration / 2
            
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(middle_time),
                "-i", video_path,
                "-vf", "scale=1920:1080,unsharp=5:5:1.0:5:5:0.0",
                "-vframes", "1",
                "-q:v", "2",
                thumbnail_path
            ]
            subprocess.run(cmd, check=True)
            
    except Exception as e:
        print(f"⚠️ Advanced thumbnail generation failed, using fallback: {e}")
        # Fallback: Simple frame extraction
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-ss", "00:00:01",
            "-vframes", "1",
            "-vf", "scale=1920:1080",
            "-q:v", "2",
            thumbnail_path
        ]
        subprocess.run(cmd, check=True)
    
    # Verify thumbnail was created
    if not os.path.exists(thumbnail_path):
        raise Exception("Failed to generate thumbnail")
    
    print(f"✅ Generated thumbnail: {thumbnail_path}")

def generate_enhanced_thumbnail(video_path, thumbnail_path, request_temp_dir):
    """Generate enhanced thumbnail with multiple options and text overlay"""
    if not os.path.exists(video_path):
        raise Exception(f"Video file not found: {video_path}")
    
    try:
        # Get video duration
        duration = get_video_duration(video_path)
        
        # Generate multiple candidate frames
        candidates = []
        time_points = [2.0, duration * 0.3, duration * 0.5, duration * 0.7]
        
        for i, time_point in enumerate(time_points):
            if time_point < duration:
                candidate_path = os.path.join(request_temp_dir, f"thumb_candidate_{i}.jpg")
                cmd = [
                    "ffmpeg", "-y",
                    "-ss", str(time_point),
                    "-i", video_path,
                    "-vf", "select='eq(pict_type,I)',scale=1920:1080,unsharp=5:5:1.0:5:5:0.0",
                    "-vframes", "1",
                    "-q:v", "1",  # Highest quality
                    candidate_path
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0 and os.path.exists(candidate_path):
                    candidates.append(candidate_path)
        
        # Use the best candidate (you could add logic to select the best one)
        if candidates:
            # For now, use the middle candidate
            best_candidate = candidates[len(candidates)//2]
            
            # Option 1: Direct copy of best frame
            shutil.copy2(best_candidate, thumbnail_path)
            
            # Option 2: Add enhancements (uncomment if needed)
            # enhance_cmd = [
            #     "ffmpeg", "-y",
            #     "-i", best_candidate,
            #     "-vf", "curves=all='0/0 0.5/0.58 1/1',eq=brightness=0.06:saturation=1.2",
            #     "-q:v", "1",
            #     thumbnail_path
            # ]
            # subprocess.run(enhance_cmd, check=True)
            
            # Clean up candidates
            for candidate in candidates:
                try:
                    os.remove(candidate)
                except:
                    pass
        else:
            # Fallback to simple method
            generate_thumbnail(video_path, thumbnail_path)
            
    except Exception as e:
        print(f"⚠️ Enhanced thumbnail generation failed: {e}")
        # Fallback to basic method
        generate_thumbnail(video_path, thumbnail_path)

def generate_bulletin_style_thumbnail(video_path, thumbnail_path, ticker_text, request_temp_dir):
    """Generate a bulletin-style thumbnail with ticker overlay"""
    if not os.path.exists(video_path):
        raise Exception(f"Video file not found: {video_path}")
    
    try:
        # First extract a good frame
        temp_frame = os.path.join(request_temp_dir, "temp_thumb_frame.jpg")
        
        # Try to get a frame when there's content (2-3 seconds in)
        cmd = [
            "ffmpeg", "-y",
            "-ss", "00:00:02.5",
            "-i", video_path,
            "-vf", "scale=1920:1080",
            "-vframes", "1",
            "-q:v", "1",
            temp_frame
        ]
        subprocess.run(cmd, check=True)
        
        # Add ticker overlay to make it look like the bulletin
        ticker_file = os.path.join(request_temp_dir, "thumb_ticker.txt")
        with open(ticker_file, "w", encoding="utf-8") as f:
            f.write(clean_hindi_input_text(ticker_text))
        
        # Create thumbnail with ticker overlay
        overlay_cmd = [
            "ffmpeg", "-y",
            "-i", temp_frame,
            "-vf", 
            # Add semi-transparent overlay at bottom
            f"drawbox=y=ih-120:color=red@0.9:width=iw:height=120:t=fill,"
            # Add ticker text
            f"drawtext=fontfile={FONT_PATH}:textfile={ticker_file}:"
            f"fontsize=48:fontcolor=white:x=(w-text_w)/2:y=h-80:"
            f"shadowcolor=black:shadowx=2:shadowy=2",
            "-q:v", "1",
            thumbnail_path
        ]
        subprocess.run(overlay_cmd, check=True)
        
        # Clean up temp files
        try:
            os.remove(temp_frame)
        except:
            pass
            
    except Exception as e:
        print(f"⚠️ Bulletin-style thumbnail failed: {e}")
        # Fallback to enhanced method
        generate_enhanced_thumbnail(video_path, thumbnail_path, request_temp_dir)

def validate_input_text(content_items):
    """Validate all text content before processing"""
    for i, item in enumerate(content_items):
        if item.segment_type == "video_with_text" and item.text:
            validation = validate_hindi_unicode(item.text)
            if not validation.get('is_valid_utf8', True):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid Unicode in content item {i}: {item.text[:50]}..."
                )
            
            if not validation.get('has_devanagari', False):
                print(f"⚠️ Warning: Content item {i} may not contain Hindi text")

def overlay_right_and_frame(bg_text_video, right_video, frame_img, output_path, temp_dir):
    """Overlay right video and frame - UPDATED FOR CONTINUOUS LOOPING"""
    bg_duration = get_video_duration(bg_text_video)
    
    temp_right_looped = os.path.join(temp_dir, "right_looped_temp.mp4")
    
    ext = os.path.splitext(right_video)[1].lower()
    
    if ext in ['.jpg', '.jpeg', '.png']:
        loop_cmd = [
            "ffmpeg", "-y",
            "-loop", "1",
            "-i", right_video,
            "-t", str(bg_duration),
            "-vf", "scale=1100:620",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            temp_right_looped
        ]
    else:
        # For videos, ensure continuous looping
        loop_cmd = [
            "ffmpeg", "-y",
            "-stream_loop", "-1",
            "-i", right_video,
            "-t", str(bg_duration + 1),  # Add buffer
            "-vf", "scale=1100:620",
            "-c:v", "libx264",
            "-preset", "fast",
            temp_right_looped
        ]
    
    subprocess.run(loop_cmd, check=True)
    
    cmd = [
        "ffmpeg", "-y",
        "-i", bg_text_video,
        "-i", temp_right_looped,
        "-i", frame_img,
        "-filter_complex",
        "[0:v][1:v]overlay=760:220:shortest=1[tmp];"
        "[tmp][2:v]overlay=0:0[out]",
        "-map", "[out]", 
        "-map", "0:a",
        "-c:v", "libx264", 
        "-c:a", "aac", 
        output_path
    ]
    subprocess.run(cmd, check=True)

def process_video_without_text(media_path, output_path):
    """Process video that doesn't need TTS"""
    cmd = [
        "ffmpeg", "-y",
        "-i", media_path,
        "-c:v", "libx264",
        "-c:a", "aac",
        output_path
    ]
    subprocess.run(cmd, check=True)

def concat_videos(video_paths, output_path):
    """Concatenate videos ensuring all audio tracks are preserved"""
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

def add_ticker_to_segment(input_video, ticker_text, output_path, temp_dir):
    """Add ticker to individual segment - only for video_with_text segments"""
    # Clean and fix ticker text properly
    fixed_ticker = clean_hindi_input_text(ticker_text)
    
    # Bottom ticker - scrolling
    repeated_bottom = (fixed_ticker + "   ") * 3
    ticker_file_bottom = os.path.join(temp_dir, "ticker_bottom.txt")
    with open(ticker_file_bottom, "w", encoding="utf-8") as f:
        f.write(repeated_bottom)
    
    # Top ticker - static
    ticker_file_top = os.path.join(temp_dir, "ticker_top.txt")
    with open(ticker_file_top, "w", encoding="utf-8") as f:
        f.write(fixed_ticker)

    cmd = [
        "ffmpeg", "-y", "-i", input_video,
        "-vf", 
        # Bottom ticker - scrolling
        f"drawbox=y=ih-60:color=red@1.0:width=iw:height=60:t=fill,"
        f"drawtext=fontfile={FONT_PATH}:textfile={ticker_file_bottom}:"
        f"fontsize=42:fontcolor=white:x=w-mod(t*200\\,w+tw):y=h-50:"
        f"shadowcolor=black:shadowx=2:shadowy=2,"
        # Top ticker - static (centered)
        f"drawbox=y=0:color=red@1.0:width=iw:height=80:t=fill,"
        f"drawtext=fontfile={FONT_PATH}:textfile={ticker_file_top}:"
        f"fontsize=48:fontcolor=white:x=(w-text_w)/2:y=15:"
        f"shadowcolor=black:shadowx=2:shadowy=2",
        "-codec:a", "copy",
        output_path
    ]

    try:
        subprocess.run(cmd, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print("❌ FFmpeg ticker command failed:")
        print(e.stderr.decode())
        raise Exception(f"Ticker overlay failed: {e.stderr.decode()}")

def add_ticker(input_video, ticker_text, output_path, temp_dir):
    """Legacy function - Add two tickers - static at top and scrolling at bottom"""
    add_ticker_to_segment(input_video, ticker_text, output_path, temp_dir)

# === API ===
@app.post("/generate")
def generate_bulletin(req: BulletinRequest):
    # Generate unique request ID
    request_id = str(uuid.uuid4())
    base_temp_dir = "temp"
    request_temp_dir = os.path.join(base_temp_dir, f"request_{request_id}")
    
    try:
        # Create base temp directory if it doesn't exist
        os.makedirs(base_temp_dir, exist_ok=True)
        # Create request-specific temp directory
        os.makedirs(request_temp_dir, exist_ok=True)
        print(f"🚀 Starting bulletin generation with ID: {request_id}")
        print(f"📁 Temp directory: {request_temp_dir}")
        
        # Validate input text first
        validate_input_text(req.content)
        
        # Download background and frame
        try:
            bg_path = download_file(req.background_url, "bg.mp4", request_temp_dir)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to download background video: {str(e)}")
        
        try:
            frame_path = download_file(req.frame_url, "frame.png", request_temp_dir)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to download frame image: {str(e)}")
        
        segments = []
        
        # Pre-process all media files
        all_text_media_paths = []
        total_duration_needed = 0
        text_segments_info = []
        
        # First pass: calculate total duration needed
        for i, item in enumerate(req.content):
            if item.segment_type == "video_with_text":
                try:
                    media_path = download_file(item.media_url, f"media_{i}.mp4", request_temp_dir)
                    
                    # Pre-process the media for better performance
                    processed_media_path = os.path.join(request_temp_dir, f"preprocessed_{i}.mp4")
                    preprocess_cmd = [
                        "ffmpeg", "-y",
                        "-i", media_path,
                        "-vf", "scale=1100:620,fps=25,format=yuv420p",
                        "-c:v", "libx264",
                        "-preset", "ultrafast",
                        "-crf", "23",
                        "-g", "25",  # Keyframe interval
                        "-bf", "0",  # No B-frames
                        "-movflags", "+faststart",  # Better streaming
                        processed_media_path
                    ]
                    subprocess.run(preprocess_cmd, check=True)
                    
                    all_text_media_paths.append(processed_media_path)
                except Exception as e:
                    print(f"⚠️ Failed to download media {i}: {str(e)}")
                    raise HTTPException(status_code=400, detail=f"Failed to download media file {i}: {str(e)}")
                
                # Process Hindi text into pages using improved functions
                try:
                    text_pages = process_content_item_text(item.text, max_chars_per_page=150)
                    
                    if not text_pages:
                        raise ValueError(f"No valid text pages generated from: {item.text[:50]}...")
                        
                    text_segments_info.append((i, item, processed_media_path, text_pages))
                    
                    # Calculate actual duration for each page
                    for page_idx, page_text in enumerate(text_pages):
                        # Generate TTS to get exact duration
                        temp_tts = os.path.join(request_temp_dir, f"temp_tts_{i}_{page_idx}.mp3")
                        generate_tts(page_text, req.language_name, req.language_code, temp_tts)
                        actual_duration = get_audio_duration(temp_tts)
                        total_duration_needed += actual_duration
                        os.remove(temp_tts)  # Clean up temp file
                        
                except Exception as e:
                    print(f"❌ Text processing failed for item {i}: {str(e)}")
                    debug_hindi_text(item.text)
                    raise HTTPException(status_code=400, detail=f"Text processing failed: {str(e)}")

        # Create combined media file with better looping
        combined_media_path = None
        if all_text_media_paths and total_duration_needed > 0:
            combined_media_path = os.path.join(request_temp_dir, "combined_media.mp4")
            print(f"📹 Creating continuous media loop for {total_duration_needed}s")
            create_continuous_media_loop(all_text_media_paths, total_duration_needed, combined_media_path, request_temp_dir)

        # Process all segments
        current_media_time = 0
        for i, item in enumerate(req.content):
            if item.segment_type == "video_without_text":
                # Download and process video without text - full screen, no ticker
                try:
                    media_path = download_file(item.media_url, f"media_{i}.mp4", request_temp_dir)
                    processed_video = os.path.join(request_temp_dir, f"processed_{i}.mp4")
                    process_video_without_text(media_path, processed_video)
                    segments.append(processed_video)
                    print(f"✅ Added video_without_text segment {i} (full screen)")
                except Exception as e:
                    print(f"⚠️ Failed to download/process media {i}: {str(e)}")
                    raise HTTPException(status_code=400, detail=f"Failed to process media file {i}: {str(e)}")

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
                        try:
                            print(f"📝 Processing page {page_idx + 1}/{len(text_pages)} for item {i}")
                            print(f"📄 Text: {page_text[:100]}...")
                            
                            tts_path = os.path.join(request_temp_dir, f"tts_{i}_{page_idx}.mp3")
                            generate_tts(page_text, req.language_name, req.language_code, tts_path)
                            duration = get_audio_duration(tts_path)

                            text_bg = os.path.join(request_temp_dir, f"textbg_{i}_{page_idx}.mp4")
                            overlay_clip = os.path.join(request_temp_dir, f"overlay_{i}_{page_idx}.mp4")
                            final_clip = os.path.join(request_temp_dir, f"final_{i}_{page_idx}.mp4")

                            create_text_background(bg_path, page_text, tts_path, text_bg, request_temp_dir)
                            
                            # Use the continuous media with proper timing
                            if combined_media_path and os.path.exists(combined_media_path):
                                # Method 1: Use pre-created continuous media
                                overlay_continuous_media(text_bg, combined_media_path, frame_path, 
                                                       current_media_time, duration, overlay_clip, request_temp_dir)
                                current_media_time += duration
                            else:
                                # Method 2: Create individual looped video for this segment
                                right_clip = os.path.join(request_temp_dir, f"smooth_right_{i}_{page_idx}.mp4")
                                
                                # Use the new smooth looping function
                                create_smooth_looped_video(media_path, duration + 1, right_clip)
                                
                                # Overlay with the smooth video
                                overlay_right_and_frame(text_bg, right_clip, frame_path, overlay_clip, request_temp_dir)
                            
                            # Add ticker only to video_with_text segments
                            add_ticker_to_segment(overlay_clip, req.ticker, final_clip, request_temp_dir)
                            
                            segments.append(final_clip)
                            print(f"✅ Completed page {page_idx + 1} for item {i} with ticker")
                            
                        except Exception as e:
                            print(f"❌ Failed to process page {page_idx} of item {i}: {str(e)}")
                            raise HTTPException(status_code=500, detail=f"Failed to process text page: {str(e)}")

        if not segments:
            raise HTTPException(status_code=400, detail="No segments were generated. Check your content items.")

        # First concatenate all segments in temp directory
        print("🔗 Concatenating all segments...")
        temp_concat_output = os.path.join(request_temp_dir, "temp_final_concat.mp4")
        concat_videos(segments, temp_concat_output)
        
        # Now copy the final video to the output directory
        final_filename = f"bulletin_{int(time.time())}.mp4"
        final_output = os.path.join(VIDEO_BULLETIN_DIR, final_filename)
        
        # Use ffmpeg to copy (this ensures proper codec/format)
        copy_cmd = [
            "ffmpeg", "-y",
            "-i", temp_concat_output,
            "-c", "copy",
            final_output
        ]
        subprocess.run(copy_cmd, check=True)

        # Ensure video exists before thumbnail
        if not os.path.exists(final_output):
            raise HTTPException(status_code=500, detail=f"Bulletin video file not found: {final_output}")

        # Generate ONE thumbnail with simple naming
        print("🖼️ Generating thumbnail...")
        thumbnail_filename = f"thumb_{int(time.time())}.jpg"
        thumbnail_path = os.path.join(THUMBNAIL_DIR, thumbnail_filename)
        
        # Check if thumbnail already exists to avoid duplicates
        if not os.path.exists(thumbnail_path):
            # Option 1: Basic enhanced thumbnail
            # generate_enhanced_thumbnail(final_output, thumbnail_path, request_temp_dir)
            
            # Option 2: Bulletin-style thumbnail with ticker (recommended)
            generate_bulletin_style_thumbnail(final_output, thumbnail_path, req.ticker, request_temp_dir)
            
            # Option 3: Simple thumbnail (uncomment if others don't work well)
            # generate_thumbnail(final_output, thumbnail_path)
        
        # Clean up request-specific temp directory
        cleanup_request_files(request_temp_dir)

        print("🎉 Bulletin generation completed successfully!")
        return {
            "status": "success", 
            "request_id": request_id,
            "output_path": final_output, 
            "thumbnail_path": thumbnail_path
        }

    except Exception as e:
        # Clean up request-specific temp directory even on error
        cleanup_request_files(request_temp_dir)
        print(f"💥 Error during bulletin generation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# === Test Functions ===
def test_hindi_processing():
    """Enhanced test function to verify improved Hindi text processing"""
    test_texts = [
        "आज के समाचार मे प्रधानमंत्री ने कहा कि गवर्नमेंट नई पॉलिसी लाएगी।",
        "न्यूज़ रिपोर्ट के अनुसार यह फैसला बहुत महत्वपूर्ण है।",
        "मै आपको बताना चाहता हूं कि यह वीडियो बहुत अच्छा है।",
        "हेलो दोस्तों, आज हम बात करेंगे न्यूज़ के बारे मे।",
        "आज सुबह दिल्ली मे हल्की बारिश हुई। लोगों को गर्मी से थोड़ी राहत मिली। स्कूल और ऑफिस समय पर खुले। मौसम विभाग ने बताया कि अगले दो दिन बारिश हो सकती है।"
    ]
    
    print("🧪 Testing Enhanced Hindi Text Processing:")
    print("=" * 70)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n📝 Test Case {i}:")
        print(f"Original: {text}")
        print(f"Length: {len(text)} chars")
        
        # Clean the text
        cleaned = clean_hindi_input_text(text)
        print(f"✅ Cleaned: {cleaned}")
        
        # Calculate display width
        display_width = calculate_hindi_width(cleaned)
        print(f"📏 Display Width: {display_width}")
        
        # Wrap text
        wrapped = wrap_hindi_text(cleaned, max_chars_per_line=30)
        print(f"📄 Wrapped Text (max 30 chars/line):")
        for line_num, line in enumerate(wrapped.split('\n'), 1):
            width = calculate_hindi_width(line)
            print(f"  Line {line_num} ({width:2d}): {line}")
        
        # Split into pages
        pages = process_content_item_text(text, max_chars_per_page=120)
        print(f"📚 Split into {len(pages)} page(s) (max 120 chars/page):")
        for page_num, page in enumerate(pages, 1):
            page_width = calculate_hindi_width(page)
            print(f"  Page {page_num} ({page_width:3d}): {page}")
        
        # Validate Unicode
        validation = validate_hindi_unicode(cleaned)
        print(f"🔍 Validation: UTF8={validation.get('is_valid_utf8')}, "
              f"Normalized={validation.get('is_normalized')}, "
              f"HasDevanagari={validation.get('has_devanagari')}")
        
        print("\n" + "─" * 70)

def test_bulletin_text_processing():
    """Test the bulletin generation with various Hindi texts"""
    test_cases = [
        {
            "text": "आज के समाचार मे प्रधानमंत्री ने कहा कि गवर्नमेंट नई पॉलिसी लाएगी।",
            "expected_fixes": ["में", "सरकार", "माननीय प्रधानमंत्री"]
        },
        {
            "text": "मै आपको बताना चाहता हूं कि यह न्यूज़ बहुत महत्वपूर्ण है।",
            "expected_fixes": ["मैं", "समाचार"]
        },
        {
            "text": "हेलो दोस्तों, आज हम बात करेंगे ब्रेकिंग न्यूज़ के बारे मे।",
            "expected_fixes": ["नमस्कार", "तत्काल", "समाचार", "में"]
        }
    ]
    
    print("🧪 Testing Bulletin Text Processing:")
    print("=" * 50)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i}:")
        print(f"Input: {case['text']}")
        
        # Process the text
        processed = clean_hindi_input_text(case['text'])
        print(f"Output: {processed}")
        
        # Check if expected fixes were applied
        fixes_applied = []
        fixes_missing = []
        
        for expected_fix in case['expected_fixes']:
            if expected_fix in processed:
                fixes_applied.append(expected_fix)
            else:
                fixes_missing.append(expected_fix)
        
        if fixes_applied:
            print(f"✅ Applied fixes: {', '.join(fixes_applied)}")
        if fixes_missing:
            print(f"❌ Missing fixes: {', '.join(fixes_missing)}")
        
        print("-" * 50)

# === Run with uvicorn ===
if __name__ == "__main__":
    # Test Hindi processing if run directly
    print("🚀 Starting Hindi Text Processing Tests...")
    test_hindi_processing()
    
    print("\n" + "="*70)
    test_bulletin_text_processing()
    
    print("\n🌟 All tests completed! Starting FastAPI server...")
    
    # Run the FastAPI server
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)