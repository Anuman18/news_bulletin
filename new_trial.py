import os
import shutil
import asyncio
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path
import uuid
import base64
from io import BytesIO
import math
import logging
import time
import json
import warnings
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import functools
import tempfile

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Suppress MoviePy warnings
warnings.filterwarnings("ignore", category=UserWarning, module="moviepy")
os.environ['IMAGEIO_FFMPEG_EXE'] = 'ffmpeg'  # Use system ffmpeg for better performance

# Import required libraries
try:
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    import requests
    from moviepy.audio.AudioClip import AudioArrayClip
    from moviepy.editor import (
        VideoFileClip, ImageClip, CompositeVideoClip, ColorClip,
        concatenate_videoclips, AudioFileClip, AudioClip, concatenate_audioclips,
        vfx, VideoClip, CompositeAudioClip
    )
    from gtts import gTTS
    # Performance enhancement imports
    from functools import lru_cache
    import aiofiles
    import httpx
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install: pip install fastapi uvicorn opencv-python-headless pillow numpy moviepy requests gtts aiofiles httpx")
    raise SystemExit(1)

# ==============================================================================
# DYNAMIC VOICE CONFIGURATION
# ==============================================================================

def extract_language_from_voice(voice_name: str) -> str:
    """
    Extract language code from any Google TTS voice name
    Supports any format like: hi-IN-Wavenet-D, en-US-Standard-A, etc.
    """
    if not voice_name:
        return "hi"
    
    try:
        # Handle formats like "hi-IN-Wavenet-D", "en-US-Standard-A"
        parts = voice_name.split("-")
        if len(parts) >= 2:
            lang_code = parts[0].lower()
            return lang_code
        
        # Handle simple formats like "hi", "en"
        if len(voice_name) <= 3:
            return voice_name.lower()
        
        # Fallback: try to extract first 2 characters
        return voice_name[:2].lower()
    except:
        return "hi"

def normalize_voice_to_gtts(voice_name: str) -> str:
    """
    Convert any Google Cloud TTS voice name to gTTS compatible language code
    This function works with ANY voice name without predefined mapping
    """
    if not voice_name:
        return "hi"
    
    lang_code = extract_language_from_voice(voice_name)
    
    # Map to gTTS supported languages
    gtts_mapping = {
        "hi": "hi",  # Hindi
        "en": "en",  # English
        "bn": "bn",  # Bengali
        "ta": "ta",  # Tamil
        "te": "te",  # Telugu
        "mr": "mr",  # Marathi
        "gu": "gu",  # Gujarati
        "kn": "kn",  # Kannada
        "ml": "ml",  # Malayalam
        "pa": "pa",  # Punjabi
        "ur": "ur",  # Urdu
        "as": "as",  # Assamese
        "or": "or",  # Odia
        "sa": "sa",  # Sanskrit
        "ne": "ne",  # Nepali
        "si": "si",  # Sinhala
        "my": "my",  # Myanmar
        "th": "th",  # Thai
        "vi": "vi",  # Vietnamese
        "zh": "zh",  # Chinese
        "ja": "ja",  # Japanese
        "ko": "ko",  # Korean
        "ar": "ar",  # Arabic
        "fa": "fa",  # Persian
        "tr": "tr",  # Turkish
        "ru": "ru",  # Russian
        "de": "de",  # German
        "fr": "fr",  # French
        "es": "es",  # Spanish
        "it": "it",  # Italian
        "pt": "pt",  # Portuguese
        "nl": "nl",  # Dutch
        "sv": "sv",  # Swedish
        "da": "da",  # Danish
        "no": "no",  # Norwegian
        "fi": "fi",  # Finnish
        "pl": "pl",  # Polish
        "cs": "cs",  # Czech
        "sk": "sk",  # Slovak
        "hu": "hu",  # Hungarian
        "ro": "ro",  # Romanian
        "bg": "bg",  # Bulgarian
        "hr": "hr",  # Croatian
        "sr": "sr",  # Serbian
        "sl": "sl",  # Slovenian
        "et": "et",  # Estonian
        "lv": "lv",  # Latvian
        "lt": "lt",  # Lithuanian
        "uk": "uk",  # Ukrainian
        "el": "el",  # Greek
        "he": "he",  # Hebrew
        "af": "af",  # Afrikaans
        "sw": "sw",  # Swahili
        "id": "id",  # Indonesian
        "ms": "ms",  # Malay
        "tl": "tl",  # Filipino
    }
    
    return gtts_mapping.get(lang_code, "hi")

# ==============================================================================
# PERFORMANCE CONFIGURATION
# ==============================================================================

# Logging Configuration - Optimized but still informative
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(threadName)s] - %(levelname)s - %(message)s"
)
logger = logging.getLogger("bulletin")

# Video Configuration
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
CONTENT_X = 40
CONTENT_Y = 80  # Increased to make room for top headline
CONTENT_WIDTH = 1840
CONTENT_HEIGHT = 860  # Adjusted for new Y position
TICKER_HEIGHT = 120
TICKER_Y = VIDEO_HEIGHT - TICKER_HEIGHT
LOGO_SIZE = 150
LOGO_X = VIDEO_WIDTH - LOGO_SIZE - 80
LOGO_Y = 30
TEXT_OVERLAY_X = 60
TEXT_OVERLAY_Y = 20  # Moved higher for better positioning

# Performance Settings
MAX_WORKERS = 8  # Increased from 4
PROCESS_WORKERS = 4  # For CPU-intensive tasks
RENDER_FPS = 24  # Keep original quality
RENDER_PRESET = "fast"  # Balance between speed and quality (was "medium")
RENDER_CRF = "22"  # Slightly higher CRF for faster encoding (was 20)
DOWNLOAD_TIMEOUT = 20  # Reduced from 30
CHUNK_SIZE = 32768  # Increased from 8192

# Cache settings
CACHE_SIZE = 128
REQUEST_CACHE = {}
FONT_CACHE = {}

# ==============================================================================
# FASTAPI APPLICATION
# ==============================================================================

app = FastAPI(
    title="News Bulletin Generator API - Top Headline Without Box",
    version="16.0.0",
    description="High-performance video bulletin generator with large top headline text"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pools for concurrent processing
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
process_executor = ProcessPoolExecutor(max_workers=PROCESS_WORKERS)

# Async HTTP client for faster downloads
http_client = httpx.AsyncClient(timeout=DOWNLOAD_TIMEOUT)

# ==============================================================================
# DIRECTORY STRUCTURE
# ==============================================================================

BASE_DIR = Path(__file__).parent
UPLOADS_DIR = BASE_DIR / "uploads"
VIDEO_BULLETIN_DIR = UPLOADS_DIR / "video-bulletin"
THUMBNAILS_DIR = UPLOADS_DIR / "thumbnails"
TEMP_BASE_DIR = UPLOADS_DIR / "temp"
FONTS_DIR = BASE_DIR / "fonts"
JSON_LOGS_DIR = UPLOADS_DIR / "json-logs"

# Create all required directories
for directory in [UPLOADS_DIR, VIDEO_BULLETIN_DIR, THUMBNAILS_DIR, TEMP_BASE_DIR, FONTS_DIR, JSON_LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# DATA MODELS
# ==============================================================================

class ContentSegment(BaseModel):
    """Model for video content segments"""
    segment_type: str  # intro, main_news, outro
    media_url: Optional[str] = None
    frame_url: Optional[str] = None
    text: Optional[str] = None
    top_headline: Optional[str] = None
    bottom_headline: Optional[str] = None
    duration: Optional[float] = None

class BulletinData(BaseModel):
    """Model for bulletin data"""
    logo_url: str
    language_code: str = "hi-IN"
    language_name: str = "Hindi"  # This can be any Google TTS voice name
    ticker: str
    background_url: str
    story_thumbnail: Optional[str] = None
    generated_story_url: Optional[str] = None
    content: List[ContentSegment]

# ==============================================================================
# SESSION MANAGEMENT
# ==============================================================================

class SessionManager:
    """Manages unique sessions for concurrent processing"""
    
    @staticmethod
    def create_session_id() -> str:
        """Create unique session ID"""
        return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    @staticmethod
    def create_session_dirs(session_id: str) -> Dict[str, Path]:
        """Create session-specific directories"""
        session_temp = TEMP_BASE_DIR / session_id
        session_audio = session_temp / "audio"
        session_downloads = session_temp / "downloads"
        
        for dir_path in [session_temp, session_audio, session_downloads]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        return {
            "temp": session_temp,
            "audio": session_audio,
            "downloads": session_downloads
        }
    
    @staticmethod
    async def cleanup_session_async(session_id: str):
        """Async cleanup session-specific directories"""
        session_temp = TEMP_BASE_DIR / session_id
        if session_temp.exists():
            try:
                await asyncio.get_event_loop().run_in_executor(
                    executor, shutil.rmtree, session_temp
                )
                logger.info(f"✅ Cleaned up session: {session_id}")
            except Exception as e:
                logger.error(f"❌ Failed to cleanup session {session_id}: {e}")

# ==============================================================================
# FONT MANAGEMENT
# ==============================================================================

HINDI_FONT_URL = "https://github.com/google/fonts/raw/main/ofl/notosansdevanagari/NotoSansDevanagari%5Bwdth%2Cwght%5D.ttf"
HINDI_FONT_PATH = FONTS_DIR / "hindi-bold.ttf"
BOLD_FONT_URL = "https://github.com/google/fonts/raw/main/apache/robotocondensed/RobotoCondensed-Bold.ttf"
BOLD_FONT_PATH = FONTS_DIR / "roboto-bold.ttf"

async def setup_fonts_async():
    """Async font setup"""
    async def download_font(url: str, path: Path):
        try:
            if path.exists():
                return
            logger.info(f"📥 Downloading font: {url}")
            response = await http_client.get(url)
            response.raise_for_status()
            async with aiofiles.open(path, "wb") as f:
                await f.write(response.content)
            logger.info(f"✅ Font saved: {path.name}")
        except Exception as e:
            logger.warning(f"⚠️ Font download failed: {e}")
    
    await download_font(HINDI_FONT_URL, HINDI_FONT_PATH)
    await download_font(BOLD_FONT_URL, BOLD_FONT_PATH)

@lru_cache(maxsize=32)
def load_font(size: int) -> ImageFont.FreeTypeFont:
    """Load font with caching"""
    try:
        if HINDI_FONT_PATH.exists():
            return ImageFont.truetype(str(HINDI_FONT_PATH), size=size, encoding="utf-8")
        if BOLD_FONT_PATH.exists():
            return ImageFont.truetype(str(BOLD_FONT_PATH), size=size)
    except Exception as e:
        logger.warning(f"Font loading error: {e}")
    return ImageFont.load_default()

# ==============================================================================
# UTILITY FUNCTIONS - OPTIMIZED
# ==============================================================================

async def save_request_response_async(session_id: str, request_data: dict, response_data: dict):
    """Async save request and response data to JSON file"""
    try:
        log_data = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "request": request_data,
            "response": response_data
        }
        
        log_file = JSON_LOGS_DIR / f"{session_id}.json"
        async with aiofiles.open(log_file, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(log_data, indent=2, ensure_ascii=False))
        
        logger.info(f"📝 Saved log: {log_file.name}")
    except Exception as e:
        logger.error(f"Failed to save log: {e}")

async def download_file_async(url: str, destination: Path, session_id: str) -> Optional[Path]:
    """Async file download with caching"""
    try:
        if not url:
            return None
        
        # Check cache first
        cache_key = f"{url}_{session_id}"
        if cache_key in REQUEST_CACHE:
            return REQUEST_CACHE[cache_key]
        
        # Handle base64 data URLs
        if url.startswith("data:"):
            header, data = url.split(",", 1)
            file_data = base64.b64decode(data)
            ext = ".png" if "png" in header else ".jpg"
            file_path = destination / f"{session_id}_{uuid.uuid4().hex[:8]}{ext}"
            
            async with aiofiles.open(file_path, "wb") as f:
                await f.write(file_data)
            
            REQUEST_CACHE[cache_key] = file_path
            return file_path if file_path.exists() else None
        
        # Download from URL with streaming
        async with http_client.stream("GET", url) as response:
            response.raise_for_status()
            
            # Determine file extension
            content_type = response.headers.get("content-type", "")
            if "video" in content_type:
                ext = ".mp4"
            elif "png" in content_type:
                ext = ".png"
            elif "jpeg" in content_type or "jpg" in content_type:
                ext = ".jpg"
            else:
                ext = Path(url.split("?")[0]).suffix or ".mp4"
            
            file_path = destination / f"{session_id}_{uuid.uuid4().hex[:8]}{ext}"
            
            async with aiofiles.open(file_path, "wb") as f:
                async for chunk in response.aiter_bytes(chunk_size=CHUNK_SIZE):
                    await f.write(chunk)
            
            if file_path.exists() and file_path.stat().st_size > 0:
                logger.info(f"[{session_id}] ✅ Downloaded: {file_path.name}")
                REQUEST_CACHE[cache_key] = file_path
                return file_path
        
        return None
    except Exception as e:
        logger.error(f"[{session_id}] Download failed: {e}")
        return None

def get_video_duration_fast(video_path: Path, session_id: str) -> float:
    """Get video duration with caching"""
    try:
        # Use cv2 for faster duration detection
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        
        if fps > 0 and frame_count > 0:
            duration = frame_count / fps
            return round(duration, 2)
        
        # Fallback to moviepy
        clip = VideoFileClip(str(video_path))
        duration = clip.duration
        clip.close()
        
        if duration is None or duration <= 0:
            logger.warning(f"[{session_id}] Invalid duration, using default 10s")
            return 10.0
        
        return round(duration, 2)
    except Exception as e:
        logger.error(f"[{session_id}] Duration error: {e}")
        return 10.0

def create_silence_audio(duration: float, fps: int = 44100) -> AudioArrayClip:
    """Create silence audio clip"""
    if duration <= 0:
        duration = 0.05
    nframes = int(round(duration * fps))
    if nframes < 1:
        nframes = 1
    silent_array = np.zeros((nframes, 2), dtype=np.float32)
    return AudioArrayClip(silent_array, fps=fps)

def estimate_tts_duration(text: str) -> float:
    """
    Better TTS duration estimation based on text characteristics
    """
    if not text or not text.strip():
        return 0.0
    
    text = text.strip()
    
    # Count words and characters
    words = len(text.split())
    chars = len(text)
    
    # Hindi/Devanagari script typically takes longer
    # Estimate based on word count with minimum duration
    base_duration = words * 0.8  # 0.8 seconds per word for Hindi
    
    # Add extra time for punctuation pauses
    punctuation_count = text.count('.') + text.count(',') + text.count('!') + text.count('?')
    pause_time = punctuation_count * 0.5
    
    total_duration = base_duration + pause_time
    
    # Minimum 3 seconds, maximum based on content
    return max(3.0, min(total_duration, chars * 0.1))

async def create_tts_audio_async(text: str, voice_name: str, duration: float, 
                                 audio_dir: Path, session_id: str) -> Optional[AudioFileClip]:
    """
    Async TTS audio generation with proper duration matching
    """
    if not text or not text.strip():
        return None
    
    try:
        # Convert any voice name to gTTS compatible language
        lang = normalize_voice_to_gtts(voice_name)
        logger.info(f"[{session_id}] Using voice '{voice_name}' -> gTTS lang '{lang}'")
        
        # Estimate TTS duration more accurately
        estimated_tts_duration = estimate_tts_duration(text)
        logger.info(f"[{session_id}] Estimated TTS duration: {estimated_tts_duration:.2f}s, Target: {duration:.2f}s")
        
        audio_path = audio_dir / f"{session_id}_tts_{uuid.uuid4().hex[:8]}.mp3"
        
        # Generate TTS in executor to avoid blocking
        loop = asyncio.get_event_loop()
        tts = await loop.run_in_executor(
            executor,
            lambda: gTTS(text=text, lang=lang, slow=False)
        )
        
        # Save TTS
        await loop.run_in_executor(
            executor,
            tts.save,
            str(audio_path)
        )
        
        if not audio_path.exists() or audio_path.stat().st_size == 0:
            return None
        
        # Load TTS audio
        tts_clip = AudioFileClip(str(audio_path))
        tts_duration = tts_clip.duration
        
        logger.info(f"[{session_id}] Actual TTS duration: {tts_duration:.2f}s")
        
        # If target duration is longer than TTS, pad with silence
        if duration > tts_duration:
            silence = create_silence_audio(duration)
            final_audio = CompositeAudioClip([tts_clip, silence])
            final_audio = final_audio.set_duration(duration)
        else:
            # If TTS is longer, use the full TTS duration
            final_audio = tts_clip.set_duration(tts_duration)
        
        return final_audio
        
    except Exception as e:
        logger.error(f"[{session_id}] TTS error: {e}")
        return None

# ==============================================================================
# VISUAL OVERLAY FUNCTIONS - MODIFIED FOR NO BOX TOP HEADLINE
# ==============================================================================

def create_logo_overlay(duration: float, logo_path: Optional[Path]) -> ImageClip:
    """Create simple logo overlay without circular background"""
    try:
        overlay = Image.new("RGBA", (VIDEO_WIDTH, VIDEO_HEIGHT), (0, 0, 0, 0))
        
        # Add logo if available
        if logo_path and logo_path.exists():
            try:
                logo = Image.open(logo_path).convert("RGBA")
                
                # Resize logo to fit the area
                ratio = min(LOGO_SIZE / logo.width, LOGO_SIZE / logo.height)
                new_size = (int(logo.width * ratio), int(logo.height * ratio))
                logo = logo.resize(new_size, Image.Resampling.LANCZOS)
                
                # Position logo
                logo_x = LOGO_X + (LOGO_SIZE - new_size[0]) // 2
                logo_y = LOGO_Y + (LOGO_SIZE - new_size[1]) // 2
                overlay.paste(logo, (logo_x, logo_y), logo)
                
                logger.info(f"Logo positioned at ({logo_x}, {logo_y}) with size {new_size}")
            except Exception as e:
                logger.error(f"Logo processing error: {e}")
        
        return ImageClip(np.array(overlay), transparent=True, duration=duration)
    except Exception as e:
        logger.error(f"Logo overlay error: {e}")
        return ImageClip(np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 4), dtype=np.uint8), transparent=True, duration=duration)

def create_frame_border(frame_path: Optional[Path], duration: float) -> ImageClip:
    """Create decorative frame border"""
    try:
        if frame_path and frame_path.exists():
            img = Image.open(frame_path).convert("RGBA")
            img = img.resize((VIDEO_WIDTH, VIDEO_HEIGHT), Image.Resampling.BILINEAR)
        else:
            img = Image.new("RGBA", (VIDEO_WIDTH, VIDEO_HEIGHT), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            
            # Create simple border
            draw.rectangle(
                [CONTENT_X - 10, CONTENT_Y - 10, CONTENT_X + CONTENT_WIDTH + 10, CONTENT_Y + CONTENT_HEIGHT + 10],
                outline=(255, 215, 0, 255),
                width=8
            )
            
            draw.rectangle(
                [CONTENT_X - 5, CONTENT_Y - 5, CONTENT_X + CONTENT_WIDTH + 5, CONTENT_Y + CONTENT_HEIGHT + 5],
                outline=(255, 255, 255, 220),
                width=2
            )
        
        return ImageClip(np.array(img), transparent=True, duration=duration)
    except Exception as e:
        logger.error(f"Frame border error: {e}")
        return ImageClip(np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 4), dtype=np.uint8), transparent=True, duration=duration)

def create_text_overlay(text: str, duration: float) -> Optional[ImageClip]:
    """Create top headline text overlay - RED BLOCK in top left corner"""
    try:
        if not text or not text.strip():
            return None
        
        canvas = Image.new("RGBA", (VIDEO_WIDTH, VIDEO_HEIGHT), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        
        # Red block dimensions for top left corner - made larger and more prominent
        block_width = 400  # Increased width
        block_height = 90   # Increased height
        block_x = 15       # Closer to edge
        block_y = 15       # Closer to top
        
        # Create solid red background block with gradient effect
        # Main red background
        # draw.rectangle(
        #     [block_x, block_y, block_x + block_width, block_y + block_height],
        #     fill=(220, 20, 60, 255)  # Strong red color
        # )
        
        # # Add gradient overlay for depth
        # for i in range(block_height // 3):
        #     alpha = int(50 - (i * 30 / (block_height // 3)))
        #     color = (255, 255, 255, alpha)
        #     draw.rectangle(
        #         [block_x, block_y + i, block_x + block_width, block_y + i + 1],
        #         fill=color
        #     )
        
        # # Add shadow effect at bottom
        # for i in range(6):
        #     alpha = int(100 - (i * 15))
        #     color = (0, 0, 0, alpha)
        #     draw.rectangle(
        #         [block_x, block_y + block_height - i, block_x + block_width, block_y + block_height - i + 1],
        #         fill=color
        #     )
        
        # Add multiple borders for professional look
        # Outer white border
        # draw.rectangle(
        #     [block_x - 3, block_y - 3, block_x + block_width + 3, block_y + block_height + 3],
        #     outline=(255, 255, 255, 255),
        #     width=3
        # )
        
        # # Inner dark border
        # draw.rectangle(
        #     [block_x + 2, block_y + 2, block_x + block_width - 2, block_y + block_height - 2],
        #     outline=(150, 10, 30, 255),
        #     width=1
        # )
        
        # # Bottom accent stripe
        # draw.rectangle(
        #     [block_x, block_y + block_height - 8, block_x + block_width, block_y + block_height],
        #     fill=(255, 60, 80, 255)
        # )
        
        # # Top highlight
        # draw.rectangle(
        #     [block_x, block_y, block_x + block_width, block_y + 3],
        #     fill=(255, 100, 120, 180)
        # )
        
        # Calculate font size to fit in the block
        font_size = 72  # Start with larger size for prominence
        font = load_font(font_size)
        
        # Adjust font size to fit block width with padding
        available_width = block_width - 30  # 15px padding on each side
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Scale down font if text is too wide
        while text_width > available_width and font_size > 28:
            font_size -= 2
            font = load_font(font_size)
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        
        # Center text within the red block
        text_x = block_x + (block_width - text_width) // 2
        text_y = block_y + (block_height - text_height) // 2
        
        # Draw text with strong outline for visibility
        outline_width = 3  # Increased outline for better visibility
        
        # Black outline for contrast
        for ox in range(-outline_width, outline_width + 1):
            for oy in range(-outline_width, outline_width + 1):
                if ox != 0 or oy != 0:
                    draw.text((text_x + ox, text_y + oy), text, font=font, fill=(0, 0, 0, 255))
        
        # Yellow/gold main text for better contrast on red
        draw.text((text_x, text_y), text, font=font, fill=(255, 255, 255, 255))
        
        logger.info(f"Red block headline positioned at: ({block_x}, {block_y}) size: {block_width}x{block_height}")
        logger.info(f"Text positioned at: ({text_x}, {text_y}) with font size {font_size}")
        
        return ImageClip(np.array(canvas), transparent=True, duration=duration)
    except Exception as e:
        logger.error(f"Text overlay error: {e}")
        return None

def create_bottom_headline(headline: str, duration: float) -> Optional[ImageClip]:
    """Create bottom headline box with centered text"""
    try:
        if not headline or not headline.strip():
            return None
        
        canvas = Image.new("RGBA", (VIDEO_WIDTH, VIDEO_HEIGHT), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        
        # Box dimensions
        box_height = 90
        box_width = VIDEO_WIDTH - 300
        box_x = (VIDEO_WIDTH - box_width) // 2
        box_y = VIDEO_HEIGHT - TICKER_HEIGHT - box_height - 20
        
        # Draw background box with gradient effect
        for i in range(box_height):
            alpha = int(250 - (i * 30 / box_height))
            color = (15, 15, 20, alpha)
            draw.rectangle(
                [box_x, box_y + i, box_x + box_width, box_y + i + 1],
                fill=color
            )
        
        # Draw borders
        draw.rectangle(
            [box_x - 3, box_y - 3, box_x + box_width + 3, box_y + box_height + 3],
            outline=(255, 255, 255, 255),
            width=3
        )
        
        # Accent bars
        accent_height = 8
        draw.rectangle([box_x, box_y, box_x + 200, box_y + accent_height], fill=(220, 20, 60, 255))
        draw.rectangle([box_x + box_width - 200, box_y, box_x + box_width, box_y + accent_height], fill=(220, 20, 60, 255))
        
        # Add main headline text - CENTERED
        display_text = f" • {headline} • "
        font = load_font(48)
        
        # Calculate text position for CENTER alignment
        bbox = draw.textbbox((0, 0), display_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        if text_width > box_width - 40:
            font = load_font(40)
            bbox = draw.textbbox((0, 0), display_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        
        # CENTER the text in the box
        text_x = box_x + (box_width - text_width) // 2
        text_y = box_y + accent_height + ((box_height - accent_height - text_height) // 2)
        
        # Draw text with outline
        for ox, oy in [(-2,0), (2,0), (0,-2), (0,2), (-1,-1), (1,1), (-1,1), (1,-1)]:
            draw.text((text_x + ox, text_y + oy), display_text, font=font, fill=(0, 0, 0, 255))
        
        draw.text((text_x, text_y), display_text, font=font, fill=(255, 255, 255, 255))
        
        # # Add "समाचार" indicator on the left (unchanged)
        # samachar_font = load_font(36)
        # samachar_text = "समाचार"
        
        # samachar_x = box_x + 20
        # samachar_y = box_y + (box_height - 36) // 2
        
        # # Draw with outline
        # for ox, oy in [(-1,0), (1,0), (0,-1), (0,1)]:
        #     draw.text((samachar_x + ox, samachar_y + oy), samachar_text, font=samachar_font, fill=(0, 0, 0, 255))
        
        # draw.text((samachar_x, samachar_y), samachar_text, font=samachar_font, fill=(255, 50, 50, 255))
        
        return ImageClip(np.array(canvas), transparent=True, duration=duration)
        
    except Exception as e:
        logger.error(f"Bottom headline error: {e}")
        return None

def create_ticker_optimized(ticker_text: str, duration: float, speed: int = 150) -> VideoClip:
    """Create scrolling news ticker with BREAKING NEWS - TEXT CENTERED in box"""
    try:
        bar_width, bar_height = VIDEO_WIDTH, TICKER_HEIGHT
        font = load_font(52)
        
        # Create text strip
        text_strip = f"  ●  {ticker_text}  ●  {ticker_text}  ●  "
        
        # Pre-render text image
        dummy_img = Image.new("RGB", (10, 10))
        dummy_draw = ImageDraw.Draw(dummy_img)
        bbox = dummy_draw.textbbox((0, 0), text_strip, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        strip_width = max(text_width + VIDEO_WIDTH, VIDEO_WIDTH * 2)
        strip_height = max(text_height + 10, bar_height - 20)
        
        # Create text image once
        strip_img = Image.new("RGBA", (strip_width, strip_height), (0, 0, 0, 0))
        strip_draw = ImageDraw.Draw(strip_img)
        
        y_pos = (strip_height - text_height) // 2
        # Draw outline
        for ox, oy in [(-2,0), (2,0), (0,-2), (0,2), (-1,-1), (1,1), (-1,1), (1,-1)]:
            strip_draw.text((ox, y_pos + oy), text_strip, font=font, fill=(0, 0, 0, 255))
        strip_draw.text((0, y_pos), text_strip, font=font, fill=(255, 255, 255, 255))
        
        strip_array = np.array(strip_img)
        
        # Create BREAKING NEWS badge with CENTERED text
        badge_width = 280
        badge_height = bar_height
        badge = Image.new("RGB", (badge_width, badge_height), (220, 20, 60))
        badge_draw = ImageDraw.Draw(badge)
        
        # Add border to badge
        badge_draw.rectangle([2, 2, badge_width - 3, badge_height - 3], outline=(255, 255, 255), width=3)
        
        # Add "BREAKING NEWS" text - CENTERED within badge
        breaking_font = load_font(37)
        news_font = load_font(38)
        
        # Calculate positions for CENTER alignment
        breaking_bbox = badge_draw.textbbox((0, 0), "BREAKING", font=breaking_font)
        breaking_width = breaking_bbox[2] - breaking_bbox[0]
        breaking_x = (badge_width - breaking_width) // 2  # Center horizontally
        
        news_bbox = badge_draw.textbbox((0, 0), "NEWS", font=news_font)
        news_width = news_bbox[2] - news_bbox[0]
        news_x = (badge_width - news_width) // 2  # Center horizontally
        
        # Draw text with shadow for better visibility - CENTERED
        badge_draw.text((breaking_x + 1, 20), "BREAKING", font=breaking_font, fill=(0, 0, 0))
        badge_draw.text((breaking_x, 19), "BREAKING", font=breaking_font, fill=(255, 255, 255))
        
        badge_draw.text((news_x + 1, 50), "NEWS", font=news_font, fill=(0, 0, 0))
        badge_draw.text((news_x, 49), "NEWS", font=news_font, fill=(255, 255, 255))
        
        badge_array = np.array(badge)
        
        def make_ticker_frame(t: float):
            # Create red background
            frame = np.full((bar_height, bar_width, 3), (160, 20, 30), dtype=np.uint8)
            
            # Calculate scroll position for text (start after the badge)
            shift = int((t * speed) % strip_width)
            x_pos = badge_width - shift  # Start scrolling after the badge
            
            # Composite scrolling text
            y_pos = (bar_height - strip_height) // 2
            
            while x_pos < bar_width:
                left = max(badge_width, x_pos)  # Don't overlap with badge
                right = min(bar_width, x_pos + strip_width)
                
                if right > left:
                    src_x1 = left - x_pos
                    src_x2 = src_x1 + (right - left)
                    
                    if src_x2 <= strip_width and src_x1 >= 0:
                        text_section = strip_array[:, src_x1:src_x2, :]
                        alpha = text_section[:, :, 3:4] / 255.0
                        
                        if y_pos >= 0 and y_pos + strip_height <= bar_height:
                            frame[y_pos:y_pos + strip_height, left:right, :] = (
                                (1 - alpha) * frame[y_pos:y_pos + strip_height, left:right, :] +
                                alpha * text_section[:, :, :3]
                            ).astype(np.uint8)
                
                x_pos += strip_width
            
            # Add BREAKING NEWS badge - positioned at left corner (x=0)
            frame[0:badge_height, 0:badge_width, :] = badge_array
            
            return frame
        
        clip = VideoClip(make_ticker_frame, duration=duration)
        return clip.set_position((0, TICKER_Y))
        
    except Exception as e:
        logger.error(f"Ticker error: {e}")
        # Return simple colored bar as fallback
        return ColorClip(size=(VIDEO_WIDTH, TICKER_HEIGHT), color=(160, 20, 30), duration=duration).set_position((0, TICKER_Y))

# ==============================================================================
# VIDEO PROCESSING - OPTIMIZED WITH FIXED DURATION HANDLING
# ==============================================================================

async def process_video_segment_async(
    media_path: Optional[Path],
    text: Optional[str],
    voice_name: str,
    segment_type: str,
    session_dirs: Dict[str, Path],
    session_id: str
) -> Tuple[Optional[VideoClip], float]:
    """Process individual video segment with proper TTS duration handling"""
    try:
        logger.info(f"[{session_id}] Processing segment: {segment_type}")
        
        # Normalize segment types
        segment_type = segment_type.lower() if segment_type else ""
        is_fullscreen = segment_type in ["intro", "outro"]
        
        # Calculate TTS duration if text exists
        tts_duration = 0
        if text and text.strip():
            tts_duration = estimate_tts_duration(text)
            logger.info(f"[{session_id}] Estimated TTS duration: {tts_duration:.2f}s")
        
        if media_path and media_path.exists():
            file_ext = str(media_path).lower()
            
            # Process video files
            if file_ext.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                clip = VideoFileClip(str(media_path))
                video_duration = clip.duration if clip.duration else 10.0
                
                # For main_news segments with TTS, use TTS duration
                if segment_type == "main_news" and tts_duration > 0:
                    final_duration = tts_duration
                    
                    # If video is shorter than TTS, loop it
                    if video_duration < tts_duration:
                        loops = math.ceil(tts_duration / video_duration)
                        clip_list = [clip] * loops
                        clip = concatenate_videoclips(clip_list, method="compose")
                    
                    # Trim to exact TTS duration
                    safe_duration = min(final_duration, clip.duration - 0.1) if clip.duration > 0.1 else final_duration
                    clip = clip.subclip(0, safe_duration)
                else:
                    # For intro/outro, use original video duration
                    final_duration = video_duration
                    safe_duration = min(final_duration, clip.duration - 0.1) if clip.duration > 0.1 else final_duration
                    clip = clip.subclip(0, safe_duration)
                
                # Handle positioning
                if is_fullscreen:
                    clip = clip.resize((VIDEO_WIDTH, VIDEO_HEIGHT))
                else:
                    # Position in content area with proper scaling 
                    original_w, original_h = clip.w, clip.h
                    
                    if original_h > original_w:  # Portrait video
                        scale = min(CONTENT_HEIGHT / original_h, CONTENT_WIDTH / original_w)
                        new_w = int(original_w * scale)
                        new_h = int(original_h * scale)
                        
                        clip = clip.resize((new_w, new_h))
                        x_offset = CONTENT_X + (CONTENT_WIDTH - new_w) // 2
                        y_offset = CONTENT_Y + (CONTENT_HEIGHT - new_h) // 2
                        
                        # Add background for portrait videos
                        bg = ColorClip(size=(CONTENT_WIDTH, CONTENT_HEIGHT), color=(25, 30, 40), duration=safe_duration)
                        bg = bg.set_position((CONTENT_X, CONTENT_Y))
                        clip = clip.set_position((x_offset, y_offset))
                        clip = CompositeVideoClip([bg, clip])
                    else:  # Landscape video
                        clip = clip.resize((CONTENT_WIDTH, CONTENT_HEIGHT))
                        clip = clip.set_position((CONTENT_X, CONTENT_Y))
                
                # Handle audio for main_news segments
                if segment_type == "main_news":
                    clip = clip.without_audio()
                    if text and text.strip():
                        audio = await create_tts_audio_async(text, voice_name, safe_duration, 
                                                            session_dirs["audio"], session_id)
                        if audio:
                            clip = clip.set_audio(audio)
                            # Update duration to match audio if needed
                            if audio.duration > safe_duration:
                                safe_duration = audio.duration
                                clip = clip.set_duration(safe_duration)
                        else:
                            clip = clip.set_audio(create_silence_audio(safe_duration))
                    else:
                        clip = clip.set_audio(create_silence_audio(safe_duration))
                
                return clip, safe_duration
            
            # Process image files
            else:
                img = Image.open(media_path).convert("RGB")
                
                # For images with TTS, use TTS duration
                if text and text.strip():
                    duration = tts_duration
                else:
                    duration = 10.0
                
                if is_fullscreen:
                    img = img.resize((VIDEO_WIDTH, VIDEO_HEIGHT), Image.Resampling.LANCZOS)
                    clip = ImageClip(np.array(img), duration=duration)
                else:
                    img = img.resize((CONTENT_WIDTH, CONTENT_HEIGHT), Image.Resampling.LANCZOS)
                    clip = ImageClip(np.array(img), duration=duration)
                    clip = clip.set_position((CONTENT_X, CONTENT_Y))
                
                # Add TTS audio for images
                if text and text.strip():
                    audio = await create_tts_audio_async(text, voice_name, duration, 
                                                        session_dirs["audio"], session_id)
                    if audio:
                        clip = clip.set_audio(audio)
                        # Update duration to match audio
                        if audio.duration != duration:
                            duration = audio.duration
                            clip = clip.set_duration(duration)
                
                return clip, duration
        
        # No media - create placeholder with TTS
        if text and text.strip():
            duration = tts_duration
        else:
            duration = 10.0
        
        if is_fullscreen:
            placeholder = ColorClip(size=(VIDEO_WIDTH, VIDEO_HEIGHT), color=(40, 45, 55), duration=duration)
        else:
            placeholder = ColorClip(size=(CONTENT_WIDTH, CONTENT_HEIGHT), color=(40, 45, 55), duration=duration)
            placeholder = placeholder.set_position((CONTENT_X, CONTENT_Y))
        
        if text and text.strip():
            audio = await create_tts_audio_async(text, voice_name, duration, 
                                                session_dirs["audio"], session_id)
            if audio:
                placeholder = placeholder.set_audio(audio)
                # Update duration to match audio
                if audio.duration != duration:
                    duration = audio.duration
                    placeholder = placeholder.set_duration(duration)
        
        return placeholder, duration
        
    except Exception as e:
        logger.error(f"[{session_id}] Segment processing error: {e}")
        # Return fallback clip
        fallback = ColorClip(size=(CONTENT_WIDTH, CONTENT_HEIGHT), color=(40, 45, 55), duration=10.0)
        return fallback.set_position((CONTENT_X, CONTENT_Y)), 10.0

# ==============================================================================
# MAIN BULLETIN PROCESSING - OPTIMIZED
# ==============================================================================

async def process_bulletin_optimized(bulletin_data: BulletinData, session_id: str) -> Dict[str, Any]:
    """Optimized bulletin processing with BIG TEXT top headline (no box)"""
    
    session_dirs = None
    clips_to_close = []
    start_time = time.time()
    
    try:
        # Create session directories
        session_dirs = SessionManager.create_session_dirs(session_id)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"[{session_id}] 🎬 PROCESSING BULLETIN - BIG TEXT VERSION")
        logger.info(f"[{session_id}] 🗣️ Voice: {bulletin_data.language_name}")
        logger.info(f"[{session_id}] 📁 Session: {session_id}")
        logger.info(f"{'='*80}\n")
        
        # Prepare all download tasks
        download_tasks = []
        download_map = {}
        
        # Background and logo
        download_tasks.append(("background", download_file_async(
            bulletin_data.background_url, session_dirs["downloads"], session_id)))
        download_tasks.append(("logo", download_file_async(
            bulletin_data.logo_url, session_dirs["downloads"], session_id)))
        
        # Frame URL (get first one found)
        frame_url = None
        for segment in bulletin_data.content:
            if segment.frame_url:
                frame_url = segment.frame_url
                break
        
        if frame_url:
            download_tasks.append(("frame", download_file_async(
                frame_url, session_dirs["downloads"], session_id)))
        
        # Media files for segments
        for idx, segment in enumerate(bulletin_data.content):
            if segment.media_url:
                download_tasks.append((f"media_{idx}", download_file_async(
                    segment.media_url, session_dirs["downloads"], session_id)))
        
        # Execute all downloads in parallel
        logger.info(f"[{session_id}] 📥 Downloading {len(download_tasks)} resources in parallel...")
        download_results = await asyncio.gather(*[task[1] for task in download_tasks])
        
        # Map results
        for (key, _), result in zip(download_tasks, download_results):
            download_map[key] = result
        
        background_path = download_map.get("background")
        logo_path = download_map.get("logo")
        frame_path = download_map.get("frame")
        
        # Process segments data with proper duration calculation
        segment_data = []
        total_duration = 0.0
        
        logger.info(f"[{session_id}] 📊 Processing {len(bulletin_data.content)} segments...")
        
        for idx, segment in enumerate(bulletin_data.content):
            logger.info(f"\n[{session_id}] Segment {idx+1}:")
            logger.info(f"  Type: {segment.segment_type}")
            logger.info(f"  Has text: {bool(segment.text)}")
            logger.info(f"  Has media: {bool(segment.media_url)}")
            
            # Get media path
            media_path = download_map.get(f"media_{idx}")
            
            # Calculate duration based on content type
            if segment.duration:
                duration = segment.duration
            elif segment.text and segment.text.strip():
                # For segments with text, estimate TTS duration
                duration = estimate_tts_duration(segment.text)
                logger.info(f"  Estimated TTS duration: {duration:.2f}s")
            elif media_path and str(media_path).lower().endswith((".mp4", ".avi", ".mov")):
                duration = await asyncio.get_event_loop().run_in_executor(
                    executor, get_video_duration_fast, media_path, session_id
                )
                logger.info(f"  Video duration: {duration:.2f}s")
            else:
                duration = 10.0
            
            segment_data.append({
                "segment": segment,
                "media_path": media_path,
                "duration": duration
            })
            total_duration += duration
            logger.info(f"  Final duration: {duration:.2f}s")
        
        logger.info(f"\n[{session_id}] Total estimated duration: {total_duration:.1f}s")
        
        # Create background layer
        logger.info(f"\n[{session_id}] Building layers...")
        
        if background_path and background_path.exists():
            try:
                if str(background_path).lower().endswith((".mp4", ".avi", ".mov")):
                    bg_clip = VideoFileClip(str(background_path)).resize((VIDEO_WIDTH, VIDEO_HEIGHT))
                    
                    if bg_clip.duration and bg_clip.duration < total_duration:
                        loops = math.ceil(total_duration / bg_clip.duration)
                        bg_list = [bg_clip] * loops
                        bg_clip = concatenate_videoclips(bg_list, method="compose")
                    
                    safe_duration = min(total_duration, bg_clip.duration - 0.1) if bg_clip.duration > 0.1 else total_duration
                    background_clip = bg_clip.subclip(0, safe_duration).without_audio()
                else:
                    img = Image.open(background_path).convert("RGB").resize((VIDEO_WIDTH, VIDEO_HEIGHT))
                    background_clip = ImageClip(np.array(img), duration=total_duration)
            except Exception as e:
                logger.error(f"[{session_id}] Background error: {e}")
                background_clip = ColorClip(size=(VIDEO_WIDTH, VIDEO_HEIGHT), color=(25, 30, 40), duration=total_duration)
        else:
            background_clip = ColorClip(size=(VIDEO_WIDTH, VIDEO_HEIGHT), color=(25, 30, 40), duration=total_duration)
        
        clips_to_close.append(background_clip)
        
        # Process segments in parallel
        segment_tasks = []
        for data in segment_data:
            seg = data["segment"]
            segment_tasks.append(
                process_video_segment_async(
                    media_path=data["media_path"],
                    text=seg.text,
                    voice_name=bulletin_data.language_name,
                    segment_type=seg.segment_type,
                    session_dirs=session_dirs,
                    session_id=session_id
                )
            )
        
        logger.info(f"[{session_id}] Processing segments in parallel...")
        segment_results = await asyncio.gather(*segment_tasks)
        
        # Build composite with actual durations
        all_clips = [background_clip]
        current_time = 0.0
        actual_durations = []
        
        # Add segments to timeline
        for idx, (result, data) in enumerate(zip(segment_results, segment_data)):
            clip, actual_duration = result
            
            if clip:
                clip = clip.set_start(current_time)
                clips_to_close.append(clip)
                all_clips.append(clip)
                actual_durations.append(actual_duration)
                logger.info(f"[{session_id}] Segment {idx+1} actual duration: {actual_duration:.2f}s")
            else:
                actual_durations.append(data["duration"])
            
            current_time += actual_durations[-1]
        
        # Update total duration with actual values
        total_duration = sum(actual_durations)
        logger.info(f"[{session_id}] Final total duration: {total_duration:.1f}s")
        
        # Update background duration to match
        background_clip = background_clip.set_duration(total_duration)
        
        # Add overlays for each segment
        current_time = 0.0
        
        logger.info(f"[{session_id}] Adding overlays with BIG TEXT headlines...")
        
        for idx, (data, duration) in enumerate(zip(segment_data, actual_durations)):
            seg = data["segment"]
            segment_type = seg.segment_type.lower() if seg.segment_type else ""
            
            is_fullscreen = segment_type in ["intro", "outro"]
            
            if not is_fullscreen:
                # Add frame
                if frame_path:
                    frame_clip = create_frame_border(frame_path, duration)
                    frame_clip = frame_clip.set_start(current_time)
                    clips_to_close.append(frame_clip)
                    all_clips.append(frame_clip)
                
                # Add logo
                if logo_path:
                    logo_clip = create_logo_overlay(duration, logo_path)
                    logo_clip = logo_clip.set_start(current_time)
                    clips_to_close.append(logo_clip)
                    all_clips.append(logo_clip)
                
                # Add ticker with CENTERED BREAKING NEWS text
                ticker_clip = create_ticker_optimized(bulletin_data.ticker, duration)
                ticker_clip = ticker_clip.set_start(current_time)
                clips_to_close.append(ticker_clip)
                all_clips.append(ticker_clip)
            
            # Add top headline as BIG TEXT without box
            if seg.top_headline:
                text_clip = create_text_overlay(seg.top_headline, duration)
                if text_clip:
                    text_clip = text_clip.set_start(current_time)
                    clips_to_close.append(text_clip)
                    all_clips.append(text_clip)
                    logger.info(f"[{session_id}] Added BIG TEXT top headline: {seg.top_headline}")
            
            # Add bottom headline
            if seg.bottom_headline and not is_fullscreen:
                bottom_clip = create_bottom_headline(seg.bottom_headline, duration)
                if bottom_clip:
                    bottom_clip = bottom_clip.set_start(current_time)
                    clips_to_close.append(bottom_clip)
                    all_clips.append(bottom_clip)
            
            current_time += duration
        
        # Create final composite
        logger.info(f"\n[{session_id}] Creating final composite with BIG TEXT headlines...")
        final_video = CompositeVideoClip(all_clips, size=(VIDEO_WIDTH, VIDEO_HEIGHT))
        final_video = final_video.set_duration(total_duration)
        
        # Generate unique alphanumeric ID (15 characters, lowercase)
        unique_id = ''.join([uuid.uuid4().hex[:8], uuid.uuid4().hex[:7]]).lower()
        
        # Generate thumbnail
        try:
            safe_time = min(1.0, (total_duration - 0.2) / 2) if total_duration > 0.2 else 0.1
            frame = final_video.get_frame(safe_time)
            thumbnail = Image.fromarray(frame).resize((1280, 720), Image.Resampling.LANCZOS)
        except:
            thumbnail = Image.new("RGB", (1280, 720), (25, 30, 40))
        
        thumbnail_filename = f"{unique_id}.jpg"
        thumbnail_path = THUMBNAILS_DIR / thumbnail_filename
        thumbnail.save(thumbnail_path, "JPEG", quality=90)
        
        # Render video with optimized settings
        video_filename = f"{unique_id}.mp4"
        video_path = VIDEO_BULLETIN_DIR / video_filename
        
        logger.info(f"\n[{session_id}] Rendering video with BIG TEXT layout...")
        logger.info(f"  Output: {video_filename}")
        logger.info(f"  Duration: {total_duration:.1f}s")
        logger.info(f"  Voice: {bulletin_data.language_name}")
        logger.info(f"  Layout: BIG TEXT headlines without box")
        
        # Optimized rendering parameters
        final_video.write_videofile(
            str(video_path),
            fps=RENDER_FPS,
            codec="libx264",
            audio_codec="aac",
            audio_bitrate="128k",
            preset=RENDER_PRESET,
            ffmpeg_params=[
                "-crf", RENDER_CRF,
                "-ar", "44100",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-threads", "0"
            ],
            threads=8,
            logger=None,
            verbose=False,
            temp_audiofile=str(session_dirs["temp"] / f"temp_audio_{session_id}.m4a")
        )
        
        # Cleanup
        final_video.close()
        for clip in clips_to_close:
            try:
                clip.close()
            except:
                pass
        
        await SessionManager.cleanup_session_async(session_id)
        
        # Calculate metrics
        processing_time = time.time() - start_time
        file_size_mb = video_path.stat().st_size / (1024 * 1024)
        
        logger.info(f"\n[{session_id}] ✅ SUCCESS! BIG TEXT LAYOUT APPLIED")
        logger.info(f"  File: {video_filename}")
        logger.info(f"  Size: {file_size_mb:.1f} MB")
        logger.info(f"  Duration: {total_duration:.1f}s")
        logger.info(f"  Processing time: {processing_time:.1f}s")
        logger.info(f"  Voice used: {bulletin_data.language_name}")
        logger.info(f"  ✅ Top headline: Large text without box")
        logger.info(f"  ✅ Ticker: Centered BREAKING NEWS text")
        
        return f"/video-bulletin/{video_filename}"
        
    except Exception as e:
        logger.error(f"[{session_id}] Error: {e}", exc_info=True)
        
        # Cleanup on error
        for clip in clips_to_close:
            try:
                clip.close()
            except:
                pass
        
        if session_dirs:
            await SessionManager.cleanup_session_async(session_id)
        
        raise HTTPException(status_code=500, detail=str(e))

# ==============================================================================
# API ENDPOINTS
# ==============================================================================

@app.post("/generate-bulletin")
async def generate_bulletin(request: BulletinData):
    """Generate news bulletin endpoint with BIG TEXT headlines"""
    try:
        session_id = SessionManager.create_session_id()
        logger.info(f"\n🚀 New request - Session: {session_id}")
        logger.info(f"🗣️ Voice requested: {request.language_name}")
        logger.info(f"📍 Layout: BIG TEXT headlines without box")
        
        # Process bulletin
        video_url = await process_bulletin_optimized(request, session_id)
        
        return video_url
        
    except Exception as e:
        logger.error(f"API error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/video-bulletin/{filename}")
async def get_video(filename: str):
    """Download video endpoint"""
    video_path = VIDEO_BULLETIN_DIR / filename
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(path=video_path, media_type="video/mp4", filename=filename)

@app.get("/thumbnails/{filename}")
async def get_thumbnail(filename: str):
    """Get thumbnail endpoint"""
    thumb_path = THUMBNAILS_DIR / filename
    if not thumb_path.exists():
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    return FileResponse(path=thumb_path, media_type="image/jpeg", filename=filename)

@app.get("/json-logs/{session_id}")
async def get_json_log(session_id: str):
    """Get JSON log endpoint"""
    log_path = JSON_LOGS_DIR / f"{session_id}.json"
    if not log_path.exists():
        raise HTTPException(status_code=404, detail="Log not found")
    return FileResponse(path=log_path, media_type="application/json", filename=f"{session_id}.json")

@app.get("/")
async def root():
    """API information endpoint"""
    return {
        "name": "News Bulletin Generator API - Big Text Version",
        "version": "16.0.0",
        "status": "ready",
        "changes": [
            "Top headline (top_headline) now displays as LARGE TEXT without red box",
            "Text covers the red frame area with big, prominent font",
            "BREAKING NEWS text in ticker is now CENTERED in the box",
            "Better text visibility with strong outlines",
            "All other functionality remains the same"
        ],
        "features": [
            "BIG TEXT top headline without background box",
            "CENTERED BREAKING NEWS text in ticker",
            "Dynamic Google TTS voice support",
            "Proper TTS duration calculation",
            "Clean logo overlay without background",
            "Session isolation",
            "Auto-cleanup",
            "JSON logging",
            "HD output (1920x1080)",
            "Parallel processing"
        ],
        "endpoints": {
            "POST /generate-bulletin": "Generate bulletin with BIG TEXT layout",
            "GET /video-bulletin/{filename}": "Download video",
            "GET /thumbnails/{filename}": "Get thumbnail",
            "GET /json-logs/{session_id}": "Get session log",
            "GET /status": "System status",
            "DELETE /cleanup": "Manual cleanup"
        }
    }

@app.get("/status")
async def get_status():
    """System status endpoint"""
    try:
        video_count = len(list(VIDEO_BULLETIN_DIR.glob("*.mp4")))
        thumb_count = len(list(THUMBNAILS_DIR.glob("*.jpg")))
        log_count = len(list(JSON_LOGS_DIR.glob("*.json")))
        temp_sessions = len(list(TEMP_BASE_DIR.iterdir()))
        cache_size = len(REQUEST_CACHE)
        
        return {
            "status": "operational",
            "videos": video_count,
            "thumbnails": thumb_count,
            "logs": log_count,
            "active_sessions": temp_sessions,
            "cache_items": cache_size,
            "workers": MAX_WORKERS,
            "version": "16.0.0 - Big Text Version",
            "layout_changes": [
                "Top headline displayed as large text without red box",
                "Text positioned to cover red frame area",
                "BREAKING NEWS text centered in ticker box",
                "Enhanced text visibility with strong outlines"
            ]
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.delete("/cleanup")
async def cleanup():
    """Manual cleanup endpoint"""
    try:
        count = {"temp": 0, "old_files": 0, "cache": 0}
        
        # Clean temp directories
        for d in TEMP_BASE_DIR.iterdir():
            if d.is_dir():
                try:
                    shutil.rmtree(d)
                    count["temp"] += 1
                except:
                    pass
        
        # Clean old files (>24 hours)
        cutoff = datetime.now().timestamp() - 86400
        
        for path in [VIDEO_BULLETIN_DIR, THUMBNAILS_DIR, JSON_LOGS_DIR]:
            for f in path.glob("*"):
                try:
                    if f.stat().st_mtime < cutoff:
                        f.unlink()
                        count["old_files"] += 1
                except:
                    pass
        
        # Clear cache
        global REQUEST_CACHE
        count["cache"] = len(REQUEST_CACHE)
        REQUEST_CACHE.clear()
        
        return {"status": "success", "cleaned": count}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def cleanup_old_files_async():
    """Async background cleanup task"""
    try:
        cutoff = datetime.now().timestamp() - 86400
        
        loop = asyncio.get_event_loop()
        
        for path in [VIDEO_BULLETIN_DIR, THUMBNAILS_DIR, JSON_LOGS_DIR]:
            for f in path.glob("*"):
                try:
                    if f.stat().st_mtime < cutoff:
                        await loop.run_in_executor(executor, f.unlink)
                except:
                    pass
        
        for d in TEMP_BASE_DIR.iterdir():
            if d.is_dir():
                try:
                    if d.stat().st_mtime < cutoff:
                        await loop.run_in_executor(executor, shutil.rmtree, d)
                except:
                    pass
                    
    except Exception as e:
        logger.error(f"Cleanup error: {e}")

# ==============================================================================
# STARTUP & SHUTDOWN
# ==============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("\n" + "="*80)
    logger.info(" " * 10 + "NEWS BULLETIN GENERATOR v16.0 - BIG TEXT VERSION")
    logger.info("="*80)
    logger.info("\n✅ System initialized with BIG TEXT layout!")
    logger.info("🔧 LAYOUT CHANGES APPLIED:")
    logger.info("  • Top headline (top_headline) displayed as LARGE TEXT")
    logger.info("  • NO red background box - just prominent text")
    logger.info("  • Text covers the red frame area prominently")
    logger.info("  • BREAKING NEWS text CENTERED in ticker box")
    logger.info("  • Enhanced text visibility with strong outlines")
    logger.info("\n⚡ PERFORMANCE:")
    logger.info(f"  • {MAX_WORKERS} concurrent workers")
    logger.info(f"  • {RENDER_PRESET} encoding preset")
    logger.info(f"  • Dynamic Google TTS voice support")
    logger.info(f"  • Resource caching active")
    logger.info("\n📍 POSITIONING:")
    logger.info("  • Top headline: Large centered text without box")
    logger.info("  • Ticker: Centered BREAKING NEWS text")
    logger.info("  • Logo: Right side (unchanged)")
    logger.info("  • Content: Center area (unchanged)")
    
    # Setup fonts
    await setup_fonts_async()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("\n🛑 Shutting down...")
    
    # Close HTTP client
    await http_client.aclose()
    
    # Clean all temp directories
    for d in TEMP_BASE_DIR.iterdir():
        if d.is_dir():
            try:
                shutil.rmtree(d)
            except:
                pass
    
    logger.info("Goodbye! 👋\n")

# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print(" " * 10 + "NEWS BULLETIN GENERATOR v16.0 - BIG TEXT VERSION")
    print("="*80)
    print("\n🔧 LAYOUT CHANGES IN THIS VERSION:")
    print("  ✅ Top Headline Big Text:")
    print("     • 'ताज़ा खबरे' (top_headline) now displayed as LARGE TEXT")
    print("     • NO red background box - clean, prominent text display")
    print("     • Text positioned to cover the red frame area")
    print("     • Auto-adjusting font size for optimal visibility")
    print("     • Strong black outline for text clarity")
    print("\n  ✅ Ticker BREAKING NEWS Centered:")
    print("     • BREAKING NEWS text is now CENTERED in the red box")
    print("     • Both 'BREAKING' and 'NEWS' text aligned to center")
    print("     • Box remains fixed at left position")
    print("     • Better visual balance and readability")
    print("\n📍 POSITIONING DETAILS:")
    print("  • Top Headline Text:")
    print("    - Position: Centered horizontally over red frame area")
    print("    - Y Position: 20px (covers red frame region)")
    print("    - Font Size: Auto-adjusting (starts at 80px, scales down if needed)")
    print("    - Style: White text with strong black outline")
    print("    - Background: None (transparent)")
    print("\n  • BREAKING NEWS Text:")
    print("    - Position: Centered within the red badge")
    print("    - Badge Position: Left corner (x=0)")
    print("    - Text Alignment: Center-aligned within 280px badge")
    print("    - Style: White text on red background")
    print("\n🗣️ VOICE SUPPORT:")
    print("  • Accepts any Google TTS voice name")
    print("  • Examples: hi-IN-Chirp3-HD-Achird, en-US-Standard-A")
    print("  • Automatic language detection from voice")
    print("  • Fallback to gTTS compatible languages")
    print("\n⚡ PERFORMANCE FEATURES:")
    print("  • Parallel resource downloads")
    print("  • Async processing throughout")
    print("  • Resource caching system")
    print("  • Optimized video encoding")
    print("  • Multi-core utilization")
    print("\n📊 VISUAL ELEMENTS:")
    print("  • Large prominent top headline text (no box)")
    print("  • Centered BREAKING NEWS in ticker")
    print("  • Clean logo overlay (right side)")
    print("  • Enhanced Hindi text display")
    print("  • Frame borders and decorations")
    print("  • Multi-language TTS support")
    print("\n📁 Directory Structure:")
    print("  uploads/")
    print("  ├── video-bulletin/    # Generated videos")
    print("  ├── thumbnails/        # Video thumbnails")
    print("  ├── temp/             # Temporary files (auto-cleaned)")
    print("  └── json-logs/        # Request/response logs")
    print("\n🎵 Voice Examples:")
    print("  • Hindi: hi-IN-Chirp3-HD-Achird, hi-IN-Wavenet-D")
    print("  • English: en-IN-Standard-B, en-US-Wavenet-A")
    print("  • Bengali: bn-IN-Wavenet-A, bn-IN-Neural2-B")
    print("  • Tamil: ta-IN-Standard-A, ta-IN-Wavenet-B")
    print("  • Any Google Cloud TTS voice supported!")
    print("\n🚀 Starting server at http://localhost:8000")
    print("📝 API docs at http://localhost:8000/docs")
    print("🔧 Version 16.0 with BIG TEXT layout!")
    print("="*80 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")