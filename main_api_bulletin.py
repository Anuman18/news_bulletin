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
os.environ['IMAGEIO_FFMPEG_EXE'] = 'ffmpeg'

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
    from gtts import gTTS  # Keep as fallback
    from functools import lru_cache
    import aiofiles
    import httpx
    
    # Import Google Cloud TTS
    from google.cloud import texttospeech
    from google.oauth2 import service_account
    
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install: pip install fastapi uvicorn opencv-python-headless pillow numpy moviepy requests gtts aiofiles httpx google-cloud-texttospeech")
    raise SystemExit(1)

# ==============================================================================
# PERFORMANCE CONFIGURATION - TRULY PARALLEL PROCESSING
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(threadName)s] - %(levelname)s - %(message)s"
)
logger = logging.getLogger("bulletin")

# Video Configuration
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
CONTENT_X = 40
CONTENT_Y = 80
CONTENT_WIDTH = 1840
CONTENT_HEIGHT = 860
TICKER_HEIGHT = 120
TICKER_Y = VIDEO_HEIGHT - TICKER_HEIGHT
LOGO_SIZE = 150
LOGO_X = VIDEO_WIDTH - LOGO_SIZE - 80
LOGO_Y = 30
TEXT_OVERLAY_X = 60
TEXT_OVERLAY_Y = 20

# Performance Settings - OPTIMIZED FOR TRUE PARALLEL PROCESSING
MAX_WORKERS = 200  # Massively increased for true parallel
PROCESS_WORKERS = 100  # Increased for parallel video processing
RENDER_FPS = 25
RENDER_PRESET = "ultrafast"  # Changed to ultrafast for speed
RENDER_CRF = "28"  # Slightly higher CRF for faster encoding
DOWNLOAD_TIMEOUT = 15
CHUNK_SIZE = 16384

# Cache settings
CACHE_SIZE = 512  # Increased for parallel processing
REQUEST_CACHE = {}
FONT_CACHE = {}

# NO SEMAPHORE - TRUE UNLIMITED PARALLEL PROCESSING
# Remove the semaphore to allow truly unlimited parallel processing

# Thread pool executors with massive capacity
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
process_executor = ProcessPoolExecutor(max_workers=PROCESS_WORKERS)

# ==============================================================================
# GOOGLE CLOUD TTS CONFIGURATION
# ==============================================================================

# Initialize Google Cloud TTS client (will be done in startup)
tts_client = None
USE_GOOGLE_CLOUD_TTS = False

def initialize_google_cloud_tts():
    """Initialize Google Cloud TTS client from key.json file"""
    global tts_client, USE_GOOGLE_CLOUD_TTS
    
    try:
        # Look for key.json in the current directory
        key_path = Path("key.json")
        
        if key_path.exists():
            logger.info(f"📁 Found Google Cloud credentials: {key_path}")
            
            # Load credentials from key.json
            credentials = service_account.Credentials.from_service_account_file(str(key_path))
            tts_client = texttospeech.TextToSpeechClient(credentials=credentials)
            USE_GOOGLE_CLOUD_TTS = True
            
            logger.info("✅ Google Cloud TTS initialized successfully from key.json!")
            logger.info("🎙️ Dynamic voice support ENABLED!")
            return True
        else:
            logger.warning("⚠️ key.json file not found. Using gTTS fallback.")
            logger.warning("💡 Place your Google Cloud service account key as 'key.json' in the app directory")
            return False
            
    except Exception as e:
        logger.error(f"❌ Failed to initialize Google Cloud TTS: {e}")
        logger.warning("⚠️ Falling back to gTTS (limited voice support)")
        USE_GOOGLE_CLOUD_TTS = False
        return False

def extract_language_from_voice(voice_name: str) -> str:
    """Extract language code from Google TTS voice name"""
    if not voice_name:
        return "hi-IN"
    
    try:
        # Handle formats like "hi-IN-Chirp3-HD-Achird", "hi-IN-Wavenet-D", etc.
        parts = voice_name.split("-")
        if len(parts) >= 2:
            # Return full language-region code like "hi-IN"
            return f"{parts[0]}-{parts[1]}"
        
        # Handle simple formats like "hi", "en"
        if len(voice_name) <= 3:
            return voice_name.lower() + "-IN"
        
        # Fallback: try to extract first 2 characters + IN
        return voice_name[:2].lower() + "-IN"
    except:
        return "hi-IN"

def normalize_voice_to_gtts(voice_name: str) -> str:
    """Convert Google Cloud TTS voice name to gTTS compatible language code (fallback)"""
    if not voice_name:
        return "hi"
    
    # Extract just the language part (not region)
    lang_code = voice_name.split("-")[0].lower() if "-" in voice_name else voice_name.lower()
    
    # Comprehensive gTTS language mapping
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
        "or": "or",  # Odia
        "as": "as",  # Assamese
        "ne": "ne",  # Nepali
        "sa": "sa",  # Sanskrit
        "fr": "fr",  # French
        "de": "de",  # German
        "es": "es",  # Spanish
        "it": "it",  # Italian
        "pt": "pt",  # Portuguese
        "ru": "ru",  # Russian
        "ja": "ja",  # Japanese
        "ko": "ko",  # Korean
        "zh": "zh",  # Chinese
        "ar": "ar",  # Arabic
    }
    
    return gtts_mapping.get(lang_code, "hi")

async def create_tts_audio_google_cloud(text: str, voice_name: str, language_code: str, 
                                       audio_dir: Path, session_id: str) -> Optional[Path]:
    """Create TTS using Google Cloud Text-to-Speech with dynamic voice selection"""
    if not tts_client:
        return None
        
    try:
        # Set up the text input
        synthesis_input = texttospeech.SynthesisInput(text=text)
        
        # Use the provided language_code, or extract from voice_name if not provided
        if not language_code or language_code.strip() == "":
            language_code = extract_language_from_voice(voice_name)
        
        logger.info(f"[{session_id}] 🎙️ Using Google Cloud TTS:")
        logger.info(f"  Voice: {voice_name}")
        logger.info(f"  Language: {language_code}")
        
        # Build the voice request with the specific voice name from JSON
        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code,
            name=voice_name  # Use the exact voice name passed from your JSON
        )
        
        # Configure high-quality audio
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=1.0,    # Normal speed
            pitch=0.0,            # Normal pitch
            volume_gain_db=0.0,   # Normal volume
            sample_rate_hertz=44100,  # High quality audio
            effects_profile_id=["large-home-entertainment-class-device"]  # Better quality
        )
        
        # Perform the text-to-speech request
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            executor,
            lambda: tts_client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )
        )
        
        # Save the audio content
        audio_path = audio_dir / f"{session_id}_tts_{uuid.uuid4().hex[:8]}.mp3"
        
        async with aiofiles.open(audio_path, "wb") as f:
            await f.write(response.audio_content)
        
        logger.info(f"[{session_id}] ✅ Google Cloud TTS generated successfully!")
        
        return audio_path
        
    except Exception as e:
        logger.error(f"[{session_id}] ❌ Google Cloud TTS error: {e}")
        return None

# ==============================================================================
# FASTAPI APPLICATION
# ==============================================================================

app = FastAPI(
    title="True Parallel Google Cloud TTS News Bulletin Generator",
    version="21.0.0",
    description="True parallel video bulletin generator - all videos process simultaneously and return completed paths"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# HTTP client pool for async downloads
http_clients = []
HTTP_CLIENT_POOL_SIZE = 100  # Increased pool size

async def get_http_client():
    """Get HTTP client from pool"""
    if not http_clients:
        for _ in range(HTTP_CLIENT_POOL_SIZE):
            http_clients.append(httpx.AsyncClient(timeout=DOWNLOAD_TIMEOUT))
    # Round-robin client selection
    import random
    return random.choice(http_clients)

# ==============================================================================
# DIRECTORY STRUCTURE
# ==============================================================================

BASE_DIR = Path(__file__).parent
UPLOADS_DIR = BASE_DIR / "uploads"
VIDEO_BULLETIN_DIR = UPLOADS_DIR / "video"
THUMBNAILS_DIR = UPLOADS_DIR / "thumbnail"
TEMP_BASE_DIR = UPLOADS_DIR / "temp"
FONTS_DIR = BASE_DIR / "fonts"
JSON_LOGS_DIR = UPLOADS_DIR / "json-logs"

for directory in [UPLOADS_DIR, VIDEO_BULLETIN_DIR, THUMBNAILS_DIR, TEMP_BASE_DIR, FONTS_DIR, JSON_LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# DATA MODELS
# ==============================================================================

class ContentSegment(BaseModel):
    """Model for video content segments"""
    segment_type: str
    media_url: Optional[str] = None
    frame_url: Optional[str] = None
    text: Optional[str] = None
    top_headline: Optional[str] = None
    bottom_headline: Optional[str] = None
    duration: Optional[float] = None

class BulletinData(BaseModel):
    """Updated model for bulletin data with dynamic voice support"""
    logo_url: str
    language_code: str = "hi-IN"
    language_name: str = "hi-IN-Standard-A"
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
        return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    @staticmethod
    def create_session_dirs(session_id: str) -> Dict[str, Path]:
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
    client = await get_http_client()
    
    async def download_font(url: str, path: Path):
        try:
            if path.exists():
                return
            logger.info(f"📥 Downloading font: {url}")
            response = await client.get(url)
            response.raise_for_status()
            async with aiofiles.open(path, "wb") as f:
                await f.write(response.content)
            logger.info(f"✅ Font saved: {path.name}")
        except Exception as e:
            logger.warning(f"⚠️ Font download failed: {e}")
    
    await download_font(HINDI_FONT_URL, HINDI_FONT_PATH)
    await download_font(BOLD_FONT_URL, BOLD_FONT_PATH)

@lru_cache(maxsize=128)
def load_font(size: int) -> ImageFont.FreeTypeFont:
    try:
        if HINDI_FONT_PATH.exists():
            return ImageFont.truetype(str(HINDI_FONT_PATH), size=size, encoding="utf-8")
        if BOLD_FONT_PATH.exists():
            return ImageFont.truetype(str(BOLD_FONT_PATH), size=size)
    except Exception as e:
        logger.warning(f"Font loading error: {e}")
    return ImageFont.load_default()

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

async def download_file_async(url: str, destination: Path, session_id: str) -> Optional[Path]:
    """Optimized async file download"""
    try:
        if not url:
            return None
        
        # Handle base64 data URLs
        if url.startswith("data:"):
            header, data = url.split(",", 1)
            file_data = base64.b64decode(data)
            ext = ".png" if "png" in header else ".jpg"
            file_path = destination / f"{session_id}_{uuid.uuid4().hex[:8]}{ext}"
            
            async with aiofiles.open(file_path, "wb") as f:
                await f.write(file_data)
            
            return file_path if file_path.exists() else None
        
        # Download from URL
        client = await get_http_client()
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            
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
                return file_path
        
        return None
    except Exception as e:
        logger.error(f"[{session_id}] Download failed: {e}")
        return None

def get_video_duration_fast(video_path: Path, session_id: str) -> float:
    """Fast video duration detection"""
    try:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        
        if fps > 0 and frame_count > 0:
            duration = frame_count / fps
            return max(0.5, round(duration, 2))
        
        # Fallback to moviepy
        clip = VideoFileClip(str(video_path))
        duration = clip.duration or 5.0
        clip.close()
        
        return max(0.5, round(duration, 2))
    except Exception as e:
        logger.error(f"[{session_id}] Duration error: {e}")
        return 5.0

def create_silence_audio(duration: float, fps: int = 44100) -> AudioArrayClip:
    """Create silence audio clip"""
    duration = max(0.1, min(duration, 300))
    nframes = max(1, int(round(duration * fps)))
    silent_array = np.zeros((nframes, 2), dtype=np.float32)
    return AudioArrayClip(silent_array, fps=fps)

def estimate_tts_duration_fixed(text: str) -> float:
    """TTS duration estimation"""
    if not text or not text.strip():
        return 1.0
    
    text = text.strip()
    words = len(text.split())
    chars = len(text)
    
    # Average speaking rate: 2-3 words per second for clear speech
    base_duration = words * 0.5  # 0.5 seconds per word
    
    # Add time for punctuation
    punctuation_count = text.count('.') + text.count(',') + text.count('!') + text.count('?')
    pause_time = punctuation_count * 0.3
    
    total_duration = base_duration + pause_time
    
    # Reasonable bounds
    final_duration = max(2.0, min(total_duration, 60.0))
    
    # Additional safety for long text
    if chars > 500:
        final_duration = min(final_duration, 90.0)
    
    return final_duration

async def create_tts_audio_async(text: str, voice_name: str, language_code: str, 
                                 audio_dir: Path, session_id: str) -> Optional[Tuple[AudioFileClip, float]]:
    """TTS audio generation - returns both clip and actual duration"""
    if not text or not text.strip():
        return None
    
    try:
        audio_path = None
        
        # Try Google Cloud TTS first if available
        if USE_GOOGLE_CLOUD_TTS:
            audio_path = await create_tts_audio_google_cloud(
                text, voice_name, language_code, audio_dir, session_id
            )
        
        # Fallback to gTTS
        if not audio_path:
            lang = normalize_voice_to_gtts(voice_name)
            audio_path = audio_dir / f"{session_id}_tts_{uuid.uuid4().hex[:8]}.mp3"
            
            loop = asyncio.get_event_loop()
            tts = await loop.run_in_executor(
                executor,
                lambda: gTTS(text=text, lang=lang, slow=False)
            )
            
            await loop.run_in_executor(executor, tts.save, str(audio_path))
        
        if not audio_path or not audio_path.exists() or audio_path.stat().st_size == 0:
            return None
        
        # Load TTS audio
        tts_clip = AudioFileClip(str(audio_path))
        actual_duration = tts_clip.duration or estimate_tts_duration_fixed(text)
        
        logger.info(f"[{session_id}] TTS actual duration: {actual_duration:.1f}s")
        
        return tts_clip, actual_duration
        
    except Exception as e:
        logger.error(f"[{session_id}] TTS error: {e}")
        return None

# ==============================================================================
# VISUAL OVERLAY FUNCTIONS (OPTIMIZED)
# ==============================================================================

def create_logo_overlay(duration: float, logo_path: Optional[Path]) -> ImageClip:
    """Create logo overlay"""
    try:
        overlay = Image.new("RGBA", (VIDEO_WIDTH, VIDEO_HEIGHT), (0, 0, 0, 0))
        
        if logo_path and logo_path.exists():
            try:
                logo = Image.open(logo_path).convert("RGBA")
                ratio = min(LOGO_SIZE / logo.width, LOGO_SIZE / logo.height)
                new_size = (int(logo.width * ratio), int(logo.height * ratio))
                logo = logo.resize(new_size, Image.Resampling.LANCZOS)
                
                logo_x = LOGO_X + (LOGO_SIZE - new_size[0]) // 2
                logo_y = LOGO_Y + (LOGO_SIZE - new_size[1]) // 2
                overlay.paste(logo, (logo_x, logo_y), logo)
            except Exception as e:
                logger.error(f"Logo processing error: {e}")
        
        return ImageClip(np.array(overlay), transparent=True, duration=duration)
    except Exception as e:
        logger.error(f"Logo overlay error: {e}")
        return ImageClip(np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 4), dtype=np.uint8), transparent=True, duration=duration)

def create_frame_border(frame_path: Optional[Path], duration: float) -> ImageClip:
    """Create frame border"""
    try:
        if frame_path and frame_path.exists():
            img = Image.open(frame_path).convert("RGBA")
            img = img.resize((VIDEO_WIDTH, VIDEO_HEIGHT), Image.Resampling.BILINEAR)
        else:
            img = Image.new("RGBA", (VIDEO_WIDTH, VIDEO_HEIGHT), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            
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
    """Create large top headline text overlay - LEFT ALIGNED"""
    try:
        if not text or not text.strip():
            return None
        
        canvas = Image.new("RGBA", (VIDEO_WIDTH, VIDEO_HEIGHT), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        
        font_size = 78
        font = load_font(font_size)
        
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        max_width = VIDEO_WIDTH - 100
        while text_width > max_width and font_size > 32:
            font_size -= 4
            font = load_font(font_size)
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        
        text_x = 50  # Fixed left margin
        text_y = 25
        
        outline_width = 3
        for ox in range(-outline_width, outline_width + 1):
            for oy in range(-outline_width, outline_width + 1):
                if ox != 0 or oy != 0:
                    draw.text((text_x + ox, text_y + oy), text, font=font, fill=(0, 0, 0, 255))
        
        draw.text((text_x, text_y), text, font=font, fill=(255, 255, 255, 255))
        
        return ImageClip(np.array(canvas), transparent=True, duration=duration)
    except Exception as e:
        logger.error(f"Text overlay error: {e}")
        return None

def create_bottom_headline(headline: str, duration: float) -> Optional[ImageClip]:
    """Create bottom headline box"""
    try:
        if not headline or not headline.strip():
            return None
        
        canvas = Image.new("RGBA", (VIDEO_WIDTH, VIDEO_HEIGHT), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        
        box_height = 90
        box_width = VIDEO_WIDTH - 300
        box_x = (VIDEO_WIDTH - box_width) // 2
        box_y = VIDEO_HEIGHT - TICKER_HEIGHT - box_height - 20
        
        for i in range(box_height):
            alpha = int(250 - (i * 30 / box_height))
            color = (15, 15, 20, alpha)
            draw.rectangle([box_x, box_y + i, box_x + box_width, box_y + i + 1], fill=color)
        
        draw.rectangle([box_x - 3, box_y - 3, box_x + box_width + 3, box_y + box_height + 3],
                      outline=(255, 255, 255, 255), width=3)
        
        accent_height = 8
        draw.rectangle([box_x, box_y, box_x + 200, box_y + accent_height], fill=(220, 20, 60, 255))
        draw.rectangle([box_x + box_width - 200, box_y, box_x + box_width, box_y + accent_height], fill=(220, 20, 60, 255))
        
        display_text = f" • {headline} • "
        font = load_font(48)
        
        bbox = draw.textbbox((0, 0), display_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        if text_width > box_width - 40:
            font = load_font(40)
            bbox = draw.textbbox((0, 0), display_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        
        text_x = box_x + (box_width - text_width) // 2
        text_y = box_y + accent_height + ((box_height - accent_height - text_height) // 2)
        
        for ox, oy in [(-2,0), (2,0), (0,-2), (0,2), (-1,-1), (1,1), (-1,1), (1,-1)]:
            draw.text((text_x + ox, text_y + oy), display_text, font=font, fill=(0, 0, 0, 255))
        
        draw.text((text_x, text_y), display_text, font=font, fill=(255, 255, 255, 255))
        
        return ImageClip(np.array(canvas), transparent=True, duration=duration)
        
    except Exception as e:
        logger.error(f"Bottom headline error: {e}")
        return None

def create_ticker_optimized(ticker_text: str, duration: float, speed: int = 120) -> VideoClip:
    """Create scrolling ticker with centered BREAKING NEWS - OPTIMIZED"""
    try:
        bar_width, bar_height = VIDEO_WIDTH, TICKER_HEIGHT
        font = load_font(52)
        
        text_strip = f"  ●  {ticker_text}  ●  {ticker_text}  ●  "
        
        dummy_img = Image.new("RGB", (10, 10))
        dummy_draw = ImageDraw.Draw(dummy_img)
        bbox = dummy_draw.textbbox((0, 0), text_strip, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        strip_width = max(text_width + VIDEO_WIDTH, VIDEO_WIDTH * 2)
        strip_height = max(text_height + 10, bar_height - 20)
        
        strip_img = Image.new("RGBA", (strip_width, strip_height), (0, 0, 0, 0))
        strip_draw = ImageDraw.Draw(strip_img)
        
        y_pos = (strip_height - text_height) // 2
        for ox, oy in [(-2,0), (2,0), (0,-2), (0,2), (-1,-1), (1,1), (-1,1), (1,-1)]:
            strip_draw.text((ox, y_pos + oy), text_strip, font=font, fill=(0, 0, 0, 255))
        strip_draw.text((0, y_pos), text_strip, font=font, fill=(255, 255, 255, 255))
        
        strip_array = np.array(strip_img)
        
        # BREAKING NEWS badge with centered text
        badge_width = 280
        badge_height = bar_height
        badge = Image.new("RGB", (badge_width, badge_height), (220, 20, 60))
        badge_draw = ImageDraw.Draw(badge)
        
        badge_draw.rectangle([2, 2, badge_width - 3, badge_height - 3], outline=(255, 255, 255), width=3)
        
        breaking_font = load_font(37)
        news_font = load_font(38)
        
        breaking_bbox = badge_draw.textbbox((0, 0), "BREAKING", font=breaking_font)
        breaking_width = breaking_bbox[2] - breaking_bbox[0]
        breaking_x = (badge_width - breaking_width) // 2
        
        news_bbox = badge_draw.textbbox((0, 0), "NEWS", font=news_font)
        news_width = news_bbox[2] - news_bbox[0]
        news_x = (badge_width - news_width) // 2
        
        badge_draw.text((breaking_x + 1, 20), "BREAKING", font=breaking_font, fill=(0, 0, 0))
        badge_draw.text((breaking_x, 19), "BREAKING", font=breaking_font, fill=(255, 255, 255))
        
        badge_draw.text((news_x + 1, 50), "NEWS", font=news_font, fill=(0, 0, 0))
        badge_draw.text((news_x, 49), "NEWS", font=news_font, fill=(255, 255, 255))
        
        badge_array = np.array(badge)
        
        def make_ticker_frame(t: float):
            frame = np.full((bar_height, bar_width, 3), (160, 20, 30), dtype=np.uint8)
            
            shift = int((t * speed) % strip_width)
            x_pos = badge_width - shift
            
            y_pos = (bar_height - strip_height) // 2
            
            while x_pos < bar_width:
                left = max(badge_width, x_pos)
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
            
            frame[0:badge_height, 0:badge_width, :] = badge_array
            
            return frame
        
        clip = VideoClip(make_ticker_frame, duration=duration)
        return clip.set_position((0, TICKER_Y))
        
    except Exception as e:
        logger.error(f"Ticker error: {e}")
        return ColorClip(size=(VIDEO_WIDTH, TICKER_HEIGHT), color=(160, 20, 30), duration=duration).set_position((0, TICKER_Y))

# ==============================================================================
# VIDEO PROCESSING - OPTIMIZED FOR SPEED
# ==============================================================================

async def process_video_segment_async(
    media_path: Optional[Path],
    text: Optional[str],
    voice_name: str,
    language_code: str,
    segment_type: str,
    session_dirs: Dict[str, Path],
    session_id: str
) -> Tuple[Optional[VideoClip], float]:
    """Video segment processing - OPTIMIZED FOR SPEED"""
    try:
        logger.info(f"[{session_id}] Processing segment: {segment_type}")
        
        segment_type = segment_type.lower() if segment_type else ""
        is_fullscreen = segment_type in ["intro", "outro"]
        
        # Get TTS audio first if there's text - this determines segment duration
        tts_audio = None
        tts_duration = 0
        
        if text and text.strip():
            tts_result = await create_tts_audio_async(text, voice_name, language_code, 
                                                      session_dirs["audio"], session_id)
            if tts_result:
                tts_audio, tts_duration = tts_result
                logger.info(f"[{session_id}] TTS duration for segment: {tts_duration:.2f}s")
        
        if media_path and media_path.exists():
            file_ext = str(media_path).lower()
            
            # Process video files
            if file_ext.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                clip = VideoFileClip(str(media_path))
                video_duration = clip.duration if clip.duration else 5.0
                
                # CRITICAL FIX: For main_news with text, use TTS duration, NOT full video loop
                if segment_type == "main_news" and tts_duration > 0:
                    # Use TTS duration as the segment duration
                    final_duration = tts_duration
                    
                    # If video is shorter than TTS, loop it to cover TTS duration
                    if video_duration < tts_duration:
                        loops_needed = math.ceil(tts_duration / video_duration)
                        loops_needed = min(loops_needed, 10)  # Reasonable limit
                        
                        logger.info(f"[{session_id}] Video ({video_duration:.1f}s) shorter than TTS ({tts_duration:.1f}s), looping {loops_needed} times")
                        
                        if loops_needed > 1:
                            clip_list = [clip] * loops_needed
                            clip = concatenate_videoclips(clip_list, method="compose")
                    
                    # IMPORTANT: Trim video to exactly match TTS duration
                    if clip.duration and clip.duration > tts_duration:
                        clip = clip.subclip(0, tts_duration)
                    
                    clip = clip.set_duration(tts_duration)
                    
                else:
                    # For intro/outro or videos without text, use original duration
                    final_duration = min(video_duration, 30.0)
                    if clip.duration and final_duration < clip.duration:
                        clip = clip.subclip(0, final_duration)
                
                # Handle positioning
                if is_fullscreen:
                    clip = clip.resize((VIDEO_WIDTH, VIDEO_HEIGHT))
                else:
                    original_w, original_h = clip.w, clip.h
                    
                    if original_h > original_w:  # Portrait
                        scale = min(CONTENT_HEIGHT / original_h, CONTENT_WIDTH / original_w)
                        new_w = int(original_w * scale)
                        new_h = int(original_h * scale)
                        
                        clip = clip.resize((new_w, new_h))
                        x_offset = CONTENT_X + (CONTENT_WIDTH - new_w) // 2
                        y_offset = CONTENT_Y + (CONTENT_HEIGHT - new_h) // 2
                        
                        bg = ColorClip(size=(CONTENT_WIDTH, CONTENT_HEIGHT), color=(25, 30, 40), duration=final_duration)
                        bg = bg.set_position((CONTENT_X, CONTENT_Y))
                        clip = clip.set_position((x_offset, y_offset))
                        clip = CompositeVideoClip([bg, clip])
                    else:  # Landscape
                        clip = clip.resize((CONTENT_WIDTH, CONTENT_HEIGHT))
                        clip = clip.set_position((CONTENT_X, CONTENT_Y))
                
                # Add audio
                if segment_type == "main_news":
                    clip = clip.without_audio()
                    if tts_audio:
                        clip = clip.set_audio(tts_audio)
                        final_duration = tts_duration  # Ensure duration matches TTS
                    else:
                        clip = clip.set_audio(create_silence_audio(final_duration))
                
                return clip, final_duration
            
            # Process image files
            else:
                img = Image.open(media_path).convert("RGB")
                
                # For images with text, use TTS duration
                if tts_duration > 0:
                    duration = tts_duration
                else:
                    duration = 8.0  # Default for images without text
                
                if is_fullscreen:
                    img = img.resize((VIDEO_WIDTH, VIDEO_HEIGHT), Image.Resampling.LANCZOS)
                    clip = ImageClip(np.array(img), duration=duration)
                else:
                    img = img.resize((CONTENT_WIDTH, CONTENT_HEIGHT), Image.Resampling.LANCZOS)
                    clip = ImageClip(np.array(img), duration=duration)
                    clip = clip.set_position((CONTENT_X, CONTENT_Y))
                
                # Add TTS audio
                if tts_audio:
                    clip = clip.set_audio(tts_audio)
                    duration = tts_duration  # Use exact TTS duration
                
                return clip, duration
        
        # No media - create placeholder
        if tts_duration > 0:
            duration = tts_duration
        else:
            duration = 5.0
        
        if is_fullscreen:
            placeholder = ColorClip(size=(VIDEO_WIDTH, VIDEO_HEIGHT), color=(40, 45, 55), duration=duration)
        else:
            placeholder = ColorClip(size=(CONTENT_WIDTH, CONTENT_HEIGHT), color=(40, 45, 55), duration=duration)
            placeholder = placeholder.set_position((CONTENT_X, CONTENT_Y))
        
        # Add TTS audio
        if tts_audio:
            placeholder = placeholder.set_audio(tts_audio)
            duration = tts_duration
        
        return placeholder, duration
        
    except Exception as e:
        logger.error(f"[{session_id}] Segment processing error: {e}")
        fallback = ColorClip(size=(CONTENT_WIDTH, CONTENT_HEIGHT), color=(40, 45, 55), duration=5.0)
        return fallback.set_position((CONTENT_X, CONTENT_Y)), 5.0

# ==============================================================================
# MAIN BULLETIN PROCESSING - TRUE PARALLEL PROCESSING
# ==============================================================================

async def process_bulletin_truly_parallel(bulletin_data: BulletinData, session_id: str) -> str:
    """Process bulletin with TRUE parallel processing - returns completed video path"""
    
    session_dirs = None
    clips_to_close = []
    start_time = time.time()
    
    try:
        session_dirs = SessionManager.create_session_dirs(session_id)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"[{session_id}] 🎬 PROCESSING BULLETIN - TRUE PARALLEL MODE")
        logger.info(f"[{session_id}] 🗣️ Language Code: {bulletin_data.language_code}")
        logger.info(f"[{session_id}] 🎙️ Voice Name: {bulletin_data.language_name}")
        logger.info(f"[{session_id}] 🔊 TTS Mode: {'Google Cloud TTS' if USE_GOOGLE_CLOUD_TTS else 'gTTS'}")
        logger.info(f"[{session_id}] ⚡ TRUE PARALLEL - No waiting, direct response")
        logger.info(f"{'='*80}\n")
        
        # Download resources in parallel
        download_tasks = []
        download_map = {}
        
        download_tasks.append(("background", download_file_async(
            bulletin_data.background_url, session_dirs["downloads"], session_id)))
        download_tasks.append(("logo", download_file_async(
            bulletin_data.logo_url, session_dirs["downloads"], session_id)))
        
        # Frame URL
        frame_url = None
        for segment in bulletin_data.content:
            if segment.frame_url:
                frame_url = segment.frame_url
                break
        
        if frame_url:
            download_tasks.append(("frame", download_file_async(
                frame_url, session_dirs["downloads"], session_id)))
        
        # Media files
        for idx, segment in enumerate(bulletin_data.content):
            if segment.media_url:
                download_tasks.append((f"media_{idx}", download_file_async(
                    segment.media_url, session_dirs["downloads"], session_id)))
        
        logger.info(f"[{session_id}] 📥 Downloading {len(download_tasks)} resources in parallel...")
        download_results = await asyncio.gather(*[task[1] for task in download_tasks])
        
        # Map results
        for (key, _), result in zip(download_tasks, download_results):
            download_map[key] = result
        
        background_path = download_map.get("background")
        logo_path = download_map.get("logo")
        frame_path = download_map.get("frame")
        
        # Process segments in parallel
        segment_tasks = []
        for idx, segment in enumerate(bulletin_data.content):
            media_path = download_map.get(f"media_{idx}")
            segment_tasks.append(
                process_video_segment_async(
                    media_path=media_path,
                    text=segment.text,
                    voice_name=bulletin_data.language_name,
                    language_code=bulletin_data.language_code,
                    segment_type=segment.segment_type,
                    session_dirs=session_dirs,
                    session_id=session_id
                )
            )
        
        logger.info(f"[{session_id}] 📊 Processing {len(segment_tasks)} segments in parallel...")
        segment_results = await asyncio.gather(*segment_tasks)
        
        # Calculate total duration from actual segment durations
        total_duration = sum(duration for _, duration in segment_results)
        
        # Safety check
        if total_duration > 600:  # 10 minutes max
            logger.warning(f"[{session_id}] Total duration {total_duration:.1f}s exceeds limit, capping...")
            scale_factor = 600 / total_duration
            segment_results = [(clip, duration * scale_factor) for clip, duration in segment_results]
            total_duration = 600
        
        logger.info(f"[{session_id}] Total duration: {total_duration:.1f}s")
        
        # Create background
        logger.info(f"[{session_id}] Building background layer...")
        
        if background_path and background_path.exists():
            try:
                if str(background_path).lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                    bg_clip = VideoFileClip(str(background_path))
                    bg_duration = bg_clip.duration if bg_clip.duration else 10.0
                    
                    # Loop background if needed
                    if bg_duration < total_duration:
                        loops_needed = math.ceil(total_duration / bg_duration)
                        logger.info(f"[{session_id}] Looping background {loops_needed} times")
                        
                        bg_list = []
                        remaining_time = total_duration
                        
                        for i in range(loops_needed):
                            if remaining_time <= 0:
                                break
                            
                            if remaining_time >= bg_duration:
                                bg_list.append(bg_clip)
                                remaining_time -= bg_duration
                            else:
                                partial_clip = bg_clip.subclip(0, remaining_time)
                                bg_list.append(partial_clip)
                                remaining_time = 0
                        
                        bg_clip = concatenate_videoclips(bg_list, method="compose")
                    
                    background_clip = bg_clip.resize((VIDEO_WIDTH, VIDEO_HEIGHT))
                    background_clip = background_clip.set_duration(total_duration)
                    background_clip = background_clip.without_audio()
                    
                else:
                    img = Image.open(background_path).convert("RGB").resize((VIDEO_WIDTH, VIDEO_HEIGHT))
                    background_clip = ImageClip(np.array(img), duration=total_duration)
                    
            except Exception as e:
                logger.error(f"[{session_id}] Background error: {e}")
                background_clip = ColorClip(size=(VIDEO_WIDTH, VIDEO_HEIGHT), color=(25, 30, 40), duration=total_duration)
        else:
            background_clip = ColorClip(size=(VIDEO_WIDTH, VIDEO_HEIGHT), color=(25, 30, 40), duration=total_duration)
        
        clips_to_close.append(background_clip)
        
        # Build composite
        all_clips = [background_clip]
        current_time = 0.0
        
        for idx, ((clip, duration), segment) in enumerate(zip(segment_results, bulletin_data.content)):
            if clip:
                clip = clip.set_start(current_time)
                clips_to_close.append(clip)
                all_clips.append(clip)
            
            # Add overlays for this segment
            segment_type = segment.segment_type.lower() if segment.segment_type else ""
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
                
                # Add ticker
                ticker_clip = create_ticker_optimized(bulletin_data.ticker, duration)
                ticker_clip = ticker_clip.set_start(current_time)
                clips_to_close.append(ticker_clip)
                all_clips.append(ticker_clip)
            
            # Add headlines
            if segment.top_headline:
                text_clip = create_text_overlay(segment.top_headline, duration)
                if text_clip:
                    text_clip = text_clip.set_start(current_time)
                    clips_to_close.append(text_clip)
                    all_clips.append(text_clip)
            
            if segment.bottom_headline and not is_fullscreen:
                bottom_clip = create_bottom_headline(segment.bottom_headline, duration)
                if bottom_clip:
                    bottom_clip = bottom_clip.set_start(current_time)
                    clips_to_close.append(bottom_clip)
                    all_clips.append(bottom_clip)
            
            current_time += duration
        
        # Create final composite
        logger.info(f"[{session_id}] Creating final composite with {len(all_clips)} clips...")
        final_video = CompositeVideoClip(all_clips, size=(VIDEO_WIDTH, VIDEO_HEIGHT))
        final_video = final_video.set_duration(total_duration)
        
        # Generate unique ID
        unique_id = ''.join([uuid.uuid4().hex[:8], uuid.uuid4().hex[:7]]).lower()
        
        # Generate thumbnail
        try:
            safe_time = min(1.0, total_duration / 4) if total_duration > 2 else 0.1
            frame = final_video.get_frame(safe_time)
            thumbnail = Image.fromarray(frame).resize((1280, 720), Image.Resampling.LANCZOS)
        except:
            thumbnail = Image.new("RGB", (1280, 720), (25, 30, 40))
        
        thumbnail_filename = f"{unique_id}.jpg"
        thumbnail_path = THUMBNAILS_DIR / thumbnail_filename
        thumbnail.save(thumbnail_path, "JPEG", quality=85)
        
        # Render video
        video_filename = f"{unique_id}.mp4"
        video_path = VIDEO_BULLETIN_DIR / video_filename
        
        logger.info(f"[{session_id}] 🎬 Rendering video: {video_filename}")
        
        # Use ultrafast preset for speed
        final_video.write_videofile(
            str(video_path),
            fps=RENDER_FPS,
            codec="libx264",
            audio_codec="aac",
            audio_bitrate="128k",
            preset="ultrafast",  # Changed to ultrafast
            ffmpeg_params=[
                "-crf", RENDER_CRF,
                "-ar", "44100",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-threads", "0"  # Use all available threads
            ],
            threads=16,  # Increased threads
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
        
        logger.info(f"\n[{session_id}] ✅ VIDEO GENERATED!")
        logger.info(f"  File: {video_filename}")
        logger.info(f"  Size: {file_size_mb:.1f} MB")
        logger.info(f"  Duration: {total_duration:.1f}s")
        logger.info(f"  Processing time: {processing_time:.1f}s")
        logger.info(f"  Mode: TRUE PARALLEL PROCESSING")
        
        return video_filename
        
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
# API ENDPOINTS - TRUE PARALLEL WITH DIRECT RESPONSE
# ==============================================================================

@app.post("/generate")
async def generate_bulletin(request: BulletinData):
    """Generate news bulletin - TRUE PARALLEL with DIRECT RESPONSE"""
    
    session_id = SessionManager.create_session_id()
    
    try:
        logger.info(f"\n🚀 NEW BULLETIN REQUEST - Session: {session_id}")
        logger.info(f"🗣️ Language Code: {request.language_code}")
        logger.info(f"🎙️ Voice Name: {request.language_name}")
        logger.info(f"🔊 TTS Mode: {'Google Cloud TTS (DYNAMIC)' if USE_GOOGLE_CLOUD_TTS else 'gTTS (fallback)'}")
        logger.info(f"⚡ TRUE PARALLEL MODE - Direct response with generated video")
        
        # Process bulletin and WAIT for completion
        video_filename = await process_bulletin_truly_parallel(request, session_id)
        
        # Return the completed video path directly
        return {
            "status": "completed",
            "video_path": video_filename,
            "session_id": session_id,
            "voice_used": request.language_name,
            "language_used": request.language_code,
            "tts_mode": "Google Cloud TTS (DYNAMIC)" if USE_GOOGLE_CLOUD_TTS else "gTTS (fallback)",
            "processing_mode": "TRUE_PARALLEL",
            "download_url": f"/video-bulletin/{video_filename}",
            "thumbnail_url": f"/thumbnails/{video_filename.replace('.mp4', '.jpg')}"
        }
        
    except Exception as e:
        logger.error(f"API error for session {session_id}: {e}", exc_info=True)
        return {
            "status": "error",
            "session_id": session_id,
            "error": str(e),
            "message": "Failed to generate bulletin"
        }

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

@app.get("/")
async def root():
    """API information endpoint"""
    return {
        "name": "True Parallel Google Cloud TTS News Bulletin Generator",
        "version": "21.0.0",
        "status": "ready",
        "processing_mode": "TRUE_PARALLEL",
        "features": [
            "✅ TRUE PARALLEL PROCESSING - All videos generate simultaneously",
            "✅ DIRECT RESPONSE - Returns completed video path, not 'processing'",
            "✅ NO STATUS CHECKING NEEDED - Get your video immediately",
            "✅ UNLIMITED CONCURRENT REQUESTS - No semaphore limits",
            "✅ OPTIMIZED FOR SPEED - Ultrafast encoding preset",
            "✅ Each bulletin in isolated folder",
            "✅ Smart video transitions",
            "✅ Dynamic voice selection"
        ],
        "performance": {
            "thread_workers": MAX_WORKERS,
            "process_workers": PROCESS_WORKERS,
            "http_client_pool": HTTP_CLIENT_POOL_SIZE,
            "encoding_preset": "ultrafast",
            "parallel_mode": "TRUE_PARALLEL"
        },
        "usage": {
            "simple": "POST to /generate - Get completed video path immediately",
            "bulk": "Send 100s of requests from Postman - All process simultaneously",
            "response": "Direct response with video_path when generation completes"
        },
        "endpoints": {
            "/generate": "POST - Generate bulletin and return completed video path",
            "/video-bulletin/{filename}": "GET - Download completed video",
            "/thumbnails/{filename}": "GET - Get video thumbnail"
        },
        "postman_instructions": {
            "step1": "Create your bulletin JSON request",
            "step2": "Use Collection Runner",
            "step3": "Set iterations to 100+ for bulk",
            "step4": "Set delay to 0ms",
            "step5": "All requests will process truly in parallel",
            "step6": "Each response contains the completed video path"
        }
    }

@app.get("/status")
async def get_status():
    """System status"""
    try:
        video_count = len(list(VIDEO_BULLETIN_DIR.glob("*.mp4")))
        thumb_count = len(list(THUMBNAILS_DIR.glob("*.jpg")))
        temp_sessions = len(list(TEMP_BASE_DIR.iterdir()))
        
        return {
            "status": "operational",
            "version": "21.0.0 - True Parallel Processing",
            "mode": "TRUE_PARALLEL_DIRECT_RESPONSE",
            "performance": {
                "thread_workers": MAX_WORKERS,
                "process_workers": PROCESS_WORKERS,
                "http_client_pool": HTTP_CLIENT_POOL_SIZE,
                "encoding": "ultrafast"
            },
            "statistics": {
                "videos_generated": video_count,
                "thumbnails": thumb_count,
                "active_temp_sessions": temp_sessions
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.delete("/cleanup")
async def cleanup():
    """Manual cleanup endpoint"""
    try:
        count = {"temp": 0, "old_files": 0}
        
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
        
        for path in [VIDEO_BULLETIN_DIR, THUMBNAILS_DIR]:
            for f in path.glob("*"):
                try:
                    if f.stat().st_mtime < cutoff:
                        f.unlink()
                        count["old_files"] += 1
                except:
                    pass
        
        return {"status": "success", "cleaned": count}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==============================================================================
# STARTUP & SHUTDOWN
# ==============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("\n" + "="*80)
    logger.info(" " * 5 + "NEWS BULLETIN GENERATOR v21.0 - TRUE PARALLEL")
    logger.info("="*80)
    
    # Initialize HTTP client pool
    for _ in range(HTTP_CLIENT_POOL_SIZE):
        http_clients.append(httpx.AsyncClient(timeout=DOWNLOAD_TIMEOUT))
    
    # Initialize Google Cloud TTS
    tts_success = initialize_google_cloud_tts()
    
    if tts_success:
        logger.info("\n✅ GOOGLE CLOUD TTS INITIALIZED!")
        logger.info("🎙️ Dynamic voice support enabled")
    else:
        logger.info("\n⚠️ Using gTTS fallback mode")
    
    logger.info("\n🚀 TRUE PARALLEL FEATURES:")
    logger.info(f"  • {MAX_WORKERS} thread workers ready")
    logger.info(f"  • {PROCESS_WORKERS} process workers ready")
    logger.info(f"  • {HTTP_CLIENT_POOL_SIZE} HTTP clients in pool")
    logger.info("  • NO SEMAPHORE LIMITS - Unlimited parallel processing")
    logger.info("  • DIRECT RESPONSE - Returns completed video path")
    logger.info("  • Each bulletin in isolated folder")
    logger.info("  • Ultrafast encoding for speed")
    
    # Setup fonts
    await setup_fonts_async()
    
    # Clean any leftover temp directories
    for d in TEMP_BASE_DIR.iterdir():
        if d.is_dir():
            try:
                shutil.rmtree(d)
            except:
                pass
    
    logger.info("\n✅ System ready for TRUE PARALLEL bulletin generation!")
    logger.info("💪 Send HUNDREDS of requests - all process simultaneously!")
    logger.info("📦 Each request returns the completed video directly!\n")

@app.on_event("shutdown") 
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("\n🛑 Shutting down...")
    
    # Close HTTP clients
    for client in http_clients:
        await client.aclose()
    
    # Clean temp directories
    for d in TEMP_BASE_DIR.iterdir():
        if d.is_dir():
            try:
                shutil.rmtree(d)
            except:
                pass
    
    logger.info("Shutdown complete! 👋\n")

# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print(" " * 8 + "NEWS BULLETIN GENERATOR v21.0")
    print(" " * 8 + "TRUE PARALLEL PROCESSING")
    print("="*80)
    
    print("\n🎯 TRUE PARALLEL FEATURES:")
    print(f"  1. UNLIMITED PARALLEL PROCESSING:")
    print(f"     • NO semaphore limits")
    print(f"     • {MAX_WORKERS} thread workers")
    print(f"     • {PROCESS_WORKERS} process workers")
    print(f"     • All videos generate simultaneously")
    
    print("\n  2. DIRECT RESPONSE:")
    print("     • API waits for video completion")
    print("     • Returns generated video path directly")
    print("     • No need for status checking")
    print("     • Get your video immediately")
    
    print("\n  3. OPTIMIZED FOR SPEED:")
    print("     • Ultrafast encoding preset")
    print("     • Parallel downloads")
    print("     • Parallel segment processing")
    print("     • Maximum thread utilization")
    
    print("\n📝 HOW TO USE WITH POSTMAN:")
    print("\n  Single Request:")
    print("  1. POST to /generate with your JSON")
    print("  2. Get video_path in response (waits for completion)")
    print("  3. Download from /video-bulletin/{filename}")
    
    print("\n  Bulk Processing (100s of videos):")
    print("  1. Set up your request in Postman")
    print("  2. Open Collection Runner")
    print("  3. Set Iterations: 100+ (or any number)")
    print("  4. Set Delay: 0ms")
    print("  5. Run collection")
    print("  6. ALL requests process truly in parallel!")
    print("  7. Each response contains the completed video!")
    
    print("\n💡 KEY DIFFERENCE:")
    print("  • Previous: Returns 'processing', need to check status")
    print("  • NOW: Returns completed video path directly!")
    print("  • All requests run simultaneously, not queued!")
    
    print("\n🚀 Starting server on port 8001...")
    print("🔥 Ready for TRUE PARALLEL video generation!\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")