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
# GOOGLE CLOUD TTS CONFIGURATION - UPDATED FOR KEY.JSON
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
        "tr": "tr",  # Turkish
        "th": "th",  # Thai
        "vi": "vi",  # Vietnamese
        "id": "id",  # Indonesian
        "ms": "ms",  # Malay
        "tl": "tl",  # Filipino
        "sw": "sw",  # Swahili
        "af": "af",  # Afrikaans
        "sq": "sq",  # Albanian
        "am": "am",  # Amharic
        "hy": "hy",  # Armenian
        "az": "az",  # Azerbaijani
        "eu": "eu",  # Basque
        "be": "be",  # Belarusian
        "bs": "bs",  # Bosnian
        "bg": "bg",  # Bulgarian
        "ca": "ca",  # Catalan
        "ceb": "ceb",  # Cebuano
        "ny": "ny",  # Chichewa
        "co": "co",  # Corsican
        "hr": "hr",  # Croatian
        "cs": "cs",  # Czech
        "da": "da",  # Danish
        "nl": "nl",  # Dutch
        "eo": "eo",  # Esperanto
        "et": "et",  # Estonian
        "fi": "fi",  # Finnish
        "fy": "fy",  # Frisian
        "gl": "gl",  # Galician
        "ka": "ka",  # Georgian
        "el": "el",  # Greek
        "ht": "ht",  # Haitian Creole
        "ha": "ha",  # Hausa
        "haw": "haw",  # Hawaiian
        "iw": "he",  # Hebrew
        "hu": "hu",  # Hungarian
        "is": "is",  # Icelandic
        "ig": "ig",  # Igbo
        "ga": "ga",  # Irish
        "jw": "jw",  # Javanese
        "kk": "kk",  # Kazakh
        "km": "km",  # Khmer
        "rw": "rw",  # Kinyarwanda
        "ky": "ky",  # Kyrgyz
        "lo": "lo",  # Lao
        "la": "la",  # Latin
        "lv": "lv",  # Latvian
        "lt": "lt",  # Lithuanian
        "lb": "lb",  # Luxembourgish
        "mk": "mk",  # Macedonian
        "mg": "mg",  # Malagasy
        "mt": "mt",  # Maltese
        "mi": "mi",  # Maori
        "mn": "mn",  # Mongolian
        "my": "my",  # Myanmar
        "no": "no",  # Norwegian
        "ps": "ps",  # Pashto
        "fa": "fa",  # Persian
        "pl": "pl",  # Polish
        "ro": "ro",  # Romanian
        "sm": "sm",  # Samoan
        "gd": "gd",  # Scots Gaelic
        "sr": "sr",  # Serbian
        "st": "st",  # Sesotho
        "sn": "sn",  # Shona
        "sd": "sd",  # Sindhi
        "si": "si",  # Sinhala
        "sk": "sk",  # Slovak
        "sl": "sl",  # Slovenian
        "so": "so",  # Somali
        "su": "su",  # Sundanese
        "sv": "sv",  # Swedish
        "tg": "tg",  # Tajik
        "tt": "tt",  # Tatar
        "tk": "tk",  # Turkmen
        "uk": "uk",  # Ukrainian
        "uz": "uz",  # Uzbek
        "cy": "cy",  # Welsh
        "xh": "xh",  # Xhosa
        "yi": "yi",  # Yiddish
        "yo": "yo",  # Yoruba
        "zu": "zu"   # Zulu
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
        logger.info(f"  Voice used: {voice_name}")
        logger.info(f"  Audio file: {audio_path.name}")
        
        return audio_path
        
    except Exception as e:
        logger.error(f"[{session_id}] ❌ Google Cloud TTS error: {e}")
        logger.error(f"  Voice attempted: {voice_name}")
        logger.error(f"  Language attempted: {language_code}")
        return None

# ==============================================================================
# PERFORMANCE CONFIGURATION
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

# Performance Settings
MAX_WORKERS = 6
PROCESS_WORKERS = 3
RENDER_FPS = 25
RENDER_PRESET = "veryfast"
RENDER_CRF = "24"
DOWNLOAD_TIMEOUT = 15
CHUNK_SIZE = 16384

# Cache settings
CACHE_SIZE = 64
REQUEST_CACHE = {}
FONT_CACHE = {}

# ==============================================================================
# FASTAPI APPLICATION
# ==============================================================================

app = FastAPI(
    title="Dynamic Google Cloud TTS News Bulletin Generator",
    version="18.0.0",
    description="High-performance video bulletin generator with dynamic Google Cloud TTS voice selection"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
process_executor = ProcessPoolExecutor(max_workers=PROCESS_WORKERS)
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

for directory in [UPLOADS_DIR, VIDEO_BULLETIN_DIR, THUMBNAILS_DIR, TEMP_BASE_DIR, FONTS_DIR, JSON_LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# DATA MODELS - UPDATED FOR DYNAMIC VOICE SELECTION
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
    language_code: str = "hi-IN"  # Language-region code (e.g., "hi-IN", "en-IN", "bn-IN")
    language_name: str = "hi-IN-Standard-A"  # Specific Google Cloud TTS voice name
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
    """Optimized async file download with caching"""
    try:
        if not url:
            return None
        
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
        
        # Download from URL
        async with http_client.stream("GET", url) as response:
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
                REQUEST_CACHE[cache_key] = file_path
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
    duration = max(0.1, min(duration, 300))  # Clamp between 0.1 and 300 seconds
    nframes = max(1, int(round(duration * fps)))
    silent_array = np.zeros((nframes, 2), dtype=np.float32)
    return AudioArrayClip(silent_array, fps=fps)

def estimate_tts_duration_fixed(text: str) -> float:
    """FIXED TTS duration estimation - prevents excessive loops"""
    if not text or not text.strip():
        return 1.0
    
    text = text.strip()
    words = len(text.split())
    chars = len(text)
    
    # More conservative estimation for Hindi/Devanagari
    # Average speaking rate: 2-3 words per second for clear speech
    base_duration = words * 0.5  # 0.5 seconds per word (2 words/sec)
    
    # Add time for punctuation (shorter pauses)
    punctuation_count = text.count('.') + text.count(',') + text.count('!') + text.count('?')
    pause_time = punctuation_count * 0.3
    
    total_duration = base_duration + pause_time
    
    # FIXED: Reasonable bounds to prevent loops
    # Minimum 2 seconds, maximum 60 seconds for any text
    final_duration = max(2.0, min(total_duration, 60.0))
    
    # Additional safety: if text is very long, cap at 90 seconds
    if chars > 500:
        final_duration = min(final_duration, 90.0)
    
    logger.info(f"TTS estimation: {words} words, {chars} chars -> {final_duration:.1f}s")
    return final_duration

async def create_tts_audio_async(text: str, voice_name: str, language_code: str, target_duration: float, 
                                 audio_dir: Path, session_id: str) -> Optional[AudioFileClip]:
    """DYNAMIC TTS audio generation with Google Cloud TTS support"""
    if not text or not text.strip():
        return None
    
    try:
        audio_path = None
        
        # Try Google Cloud TTS first if available
        if USE_GOOGLE_CLOUD_TTS:
            logger.info(f"[{session_id}] 🎙️ Using Google Cloud TTS")
            logger.info(f"  Voice: {voice_name}")
            logger.info(f"  Language: {language_code}")
            
            audio_path = await create_tts_audio_google_cloud(
                text, voice_name, language_code, audio_dir, session_id
            )
        
        # Fallback to gTTS if Google Cloud TTS fails or is not available
        if not audio_path:
            logger.info(f"[{session_id}] ⚠️ Using gTTS fallback (language only, no specific voice)")
            
            # Convert voice name to gTTS compatible language
            lang = normalize_voice_to_gtts(voice_name)
            logger.info(f"[{session_id}] Voice '{voice_name}' -> gTTS lang '{lang}'")
            
            audio_path = audio_dir / f"{session_id}_tts_{uuid.uuid4().hex[:8]}.mp3"
            
            # Generate TTS with gTTS
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
        actual_tts_duration = tts_clip.duration or estimate_tts_duration_fixed(text)
        
        logger.info(f"[{session_id}] Actual TTS: {actual_tts_duration:.1f}s")
        
        # Use the longer of TTS duration or target, but reasonable
        final_duration = max(actual_tts_duration, min(target_duration, 120.0))
        
        # If we need to extend, add silence
        if final_duration > actual_tts_duration:
            silence_duration = final_duration - actual_tts_duration
            silence = create_silence_audio(silence_duration)
            final_audio = concatenate_audioclips([tts_clip, silence])
        else:
            final_audio = tts_clip
        
        final_audio = final_audio.set_duration(final_duration)
        return final_audio
        
    except Exception as e:
        logger.error(f"[{session_id}] TTS error: {e}")
        return None

# ==============================================================================
# VISUAL OVERLAY FUNCTIONS
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
        
        # Large text positioned at top
        font_size = 78  # Start with large size
        font = load_font(font_size)
        
        # Get text dimensions
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Scale down if too wide
        max_width = VIDEO_WIDTH - 100
        while text_width > max_width and font_size > 32:
            font_size -= 4
            font = load_font(font_size)
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        
        # LEFT ALIGNED - Position text on left side instead of center
        text_x = 50  # Fixed left margin (changed from centered calculation)
        text_y = 25  # Keep same vertical position
        
        # Draw text with strong outline
        outline_width = 3
        for ox in range(-outline_width, outline_width + 1):
            for oy in range(-outline_width, outline_width + 1):
                if ox != 0 or oy != 0:
                    draw.text((text_x + ox, text_y + oy), text, font=font, fill=(0, 0, 0, 255))
        
        # Main text in white
        draw.text((text_x, text_y), text, font=font, fill=(255, 255, 255, 255))
        
        logger.info(f"Large headline LEFT-ALIGNED at: ({text_x}, {text_y}) font size {font_size}")
        
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
        
        # Background with transparency
        for i in range(box_height):
            alpha = int(250 - (i * 30 / box_height))
            color = (15, 15, 20, alpha)
            draw.rectangle([box_x, box_y + i, box_x + box_width, box_y + i + 1], fill=color)
        
        # Borders
        draw.rectangle([box_x - 3, box_y - 3, box_x + box_width + 3, box_y + box_height + 3],
                      outline=(255, 255, 255, 255), width=3)
        
        # Accent bars
        accent_height = 8
        draw.rectangle([box_x, box_y, box_x + 200, box_y + accent_height], fill=(220, 20, 60, 255))
        draw.rectangle([box_x + box_width - 200, box_y, box_x + box_width, box_y + accent_height], fill=(220, 20, 60, 255))
        
        # Text
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
        
        # Text with outline
        for ox, oy in [(-2,0), (2,0), (0,-2), (0,2), (-1,-1), (1,1), (-1,1), (1,-1)]:
            draw.text((text_x + ox, text_y + oy), display_text, font=font, fill=(0, 0, 0, 255))
        
        draw.text((text_x, text_y), display_text, font=font, fill=(255, 255, 255, 255))
        
        return ImageClip(np.array(canvas), transparent=True, duration=duration)
        
    except Exception as e:
        logger.error(f"Bottom headline error: {e}")
        return None

def create_ticker_optimized(ticker_text: str, duration: float, speed: int = 120) -> VideoClip:
    """Create scrolling ticker with centered BREAKING NEWS"""
    try:
        bar_width, bar_height = VIDEO_WIDTH, TICKER_HEIGHT
        font = load_font(52)
        
        # Create scrolling text
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
        # Outline
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
        
        # Center text in badge
        breaking_font = load_font(37)
        news_font = load_font(38)
        
        # Calculate centered positions
        breaking_bbox = badge_draw.textbbox((0, 0), "BREAKING", font=breaking_font)
        breaking_width = breaking_bbox[2] - breaking_bbox[0]
        breaking_x = (badge_width - breaking_width) // 2
        
        news_bbox = badge_draw.textbbox((0, 0), "NEWS", font=news_font)
        news_width = news_bbox[2] - news_bbox[0]
        news_x = (badge_width - news_width) // 2
        
        # Draw centered text
        badge_draw.text((breaking_x + 1, 20), "BREAKING", font=breaking_font, fill=(0, 0, 0))
        badge_draw.text((breaking_x, 19), "BREAKING", font=breaking_font, fill=(255, 255, 255))
        
        badge_draw.text((news_x + 1, 50), "NEWS", font=news_font, fill=(0, 0, 0))
        badge_draw.text((news_x, 49), "NEWS", font=news_font, fill=(255, 255, 255))
        
        badge_array = np.array(badge)
        
        def make_ticker_frame(t: float):
            # Red background
            frame = np.full((bar_height, bar_width, 3), (160, 20, 30), dtype=np.uint8)
            
            # Scrolling text position
            shift = int((t * speed) % strip_width)
            x_pos = badge_width - shift
            
            # Add scrolling text
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
            
            # Add BREAKING NEWS badge
            frame[0:badge_height, 0:badge_width, :] = badge_array
            
            return frame
        
        clip = VideoClip(make_ticker_frame, duration=duration)
        return clip.set_position((0, TICKER_Y))
        
    except Exception as e:
        logger.error(f"Ticker error: {e}")
        return ColorClip(size=(VIDEO_WIDTH, TICKER_HEIGHT), color=(160, 20, 30), duration=duration).set_position((0, TICKER_Y))

# ==============================================================================
# VIDEO PROCESSING - UPDATED FOR DYNAMIC VOICE SUPPORT
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
    """Video segment processing with dynamic Google Cloud TTS voice support"""
    try:
        logger.info(f"[{session_id}] Processing segment: {segment_type}")
        logger.info(f"  Voice: {voice_name}")
        logger.info(f"  Language: {language_code}")
        
        segment_type = segment_type.lower() if segment_type else ""
        is_fullscreen = segment_type in ["intro", "outro"]
        
        # TTS duration calculation
        tts_duration = 0
        if text and text.strip():
            tts_duration = estimate_tts_duration_fixed(text)
            logger.info(f"[{session_id}] Estimated TTS duration: {tts_duration:.2f}s")
        
        if media_path and media_path.exists():
            file_ext = str(media_path).lower()
            
            # Process video files
            if file_ext.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                clip = VideoFileClip(str(media_path))
                video_duration = clip.duration if clip.duration else 5.0
                
                # Smart duration handling
                if segment_type == "main_news" and text and text.strip():
                    # Use TTS duration for main news with text
                    final_duration = tts_duration
                    
                    # If video is much shorter, loop it REASONABLY
                    if video_duration < tts_duration and tts_duration < 120:  # Max 2 minutes
                        loops_needed = math.ceil(tts_duration / video_duration)
                        loops_needed = min(loops_needed, 5)  # Max 5 loops to prevent excessive processing
                        
                        if loops_needed > 1:
                            clip_list = [clip] * loops_needed
                            clip = concatenate_videoclips(clip_list, method="compose")
                else:
                    # For intro/outro or videos without text, use original duration
                    final_duration = min(video_duration, 30.0)  # Cap at 30 seconds for performance
                
                # Safe trim to final duration
                if clip.duration and final_duration < clip.duration:
                    safe_duration = max(0.1, min(final_duration, clip.duration - 0.1))
                    clip = clip.subclip(0, safe_duration)
                else:
                    safe_duration = final_duration
                
                # Handle positioning
                if is_fullscreen:
                    clip = clip.resize((VIDEO_WIDTH, VIDEO_HEIGHT))
                else:
                    # Content area positioning
                    original_w, original_h = clip.w, clip.h
                    
                    if original_h > original_w:  # Portrait
                        scale = min(CONTENT_HEIGHT / original_h, CONTENT_WIDTH / original_w)
                        new_w = int(original_w * scale)
                        new_h = int(original_h * scale)
                        
                        clip = clip.resize((new_w, new_h))
                        x_offset = CONTENT_X + (CONTENT_WIDTH - new_w) // 2
                        y_offset = CONTENT_Y + (CONTENT_HEIGHT - new_h) // 2
                        
                        bg = ColorClip(size=(CONTENT_WIDTH, CONTENT_HEIGHT), color=(25, 30, 40), duration=safe_duration)
                        bg = bg.set_position((CONTENT_X, CONTENT_Y))
                        clip = clip.set_position((x_offset, y_offset))
                        clip = CompositeVideoClip([bg, clip])
                    else:  # Landscape
                        clip = clip.resize((CONTENT_WIDTH, CONTENT_HEIGHT))
                        clip = clip.set_position((CONTENT_X, CONTENT_Y))
                
                # Handle audio for main_news with DYNAMIC VOICE
                if segment_type == "main_news":
                    clip = clip.without_audio()
                    if text and text.strip():
                        # Pass the dynamic voice parameters
                        audio = await create_tts_audio_async(text, voice_name, language_code, safe_duration, 
                                                            session_dirs["audio"], session_id)
                        if audio:
                            clip = clip.set_audio(audio)
                            # Update duration to match audio if reasonable
                            if audio.duration > safe_duration and audio.duration < 150:  # Max 2.5 minutes
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
                
                if text and text.strip():
                    duration = tts_duration
                else:
                    duration = 8.0  # Fixed reasonable duration for images
                
                if is_fullscreen:
                    img = img.resize((VIDEO_WIDTH, VIDEO_HEIGHT), Image.Resampling.LANCZOS)
                    clip = ImageClip(np.array(img), duration=duration)
                else:
                    img = img.resize((CONTENT_WIDTH, CONTENT_HEIGHT), Image.Resampling.LANCZOS)
                    clip = ImageClip(np.array(img), duration=duration)
                    clip = clip.set_position((CONTENT_X, CONTENT_Y))
                
                # Add TTS for images with DYNAMIC VOICE
                if text and text.strip():
                    audio = await create_tts_audio_async(text, voice_name, language_code, duration, 
                                                        session_dirs["audio"], session_id)
                    if audio:
                        clip = clip.set_audio(audio)
                        # Update duration to match audio if reasonable
                        if audio.duration != duration and audio.duration < 120:
                            duration = audio.duration
                            clip = clip.set_duration(duration)
                
                return clip, duration
        
        # No media - create placeholder
        if text and text.strip():
            duration = tts_duration
        else:
            duration = 5.0
        
        if is_fullscreen:
            placeholder = ColorClip(size=(VIDEO_WIDTH, VIDEO_HEIGHT), color=(40, 45, 55), duration=duration)
        else:
            placeholder = ColorClip(size=(CONTENT_WIDTH, CONTENT_HEIGHT), color=(40, 45, 55), duration=duration)
            placeholder = placeholder.set_position((CONTENT_X, CONTENT_Y))
        
        # Add TTS for placeholder with DYNAMIC VOICE
        if text and text.strip():
            audio = await create_tts_audio_async(text, voice_name, language_code, duration, 
                                                session_dirs["audio"], session_id)
            if audio:
                placeholder = placeholder.set_audio(audio)
                if audio.duration != duration and audio.duration < 120:
                    duration = audio.duration
                    placeholder = placeholder.set_duration(duration)
        
        return placeholder, duration
        
    except Exception as e:
        logger.error(f"[{session_id}] Segment processing error: {e}")
        fallback = ColorClip(size=(CONTENT_WIDTH, CONTENT_HEIGHT), color=(40, 45, 55), duration=5.0)
        return fallback.set_position((CONTENT_X, CONTENT_Y)), 5.0

# ==============================================================================
# MAIN BULLETIN PROCESSING - WITH DYNAMIC GOOGLE CLOUD TTS
# ==============================================================================

# ==============================================================================
# MAIN BULLETIN PROCESSING - FIXED BACKGROUND DURATION ISSUE
# ==============================================================================

async def process_bulletin_optimized(bulletin_data: BulletinData, session_id: str) -> str:
    """Process bulletin with dynamic Google Cloud TTS voice selection - FIXED BACKGROUND"""
    
    session_dirs = None
    clips_to_close = []
    start_time = time.time()
    
    try:
        session_dirs = SessionManager.create_session_dirs(session_id)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"[{session_id}] 🎬 PROCESSING BULLETIN - DYNAMIC VOICE EDITION")
        logger.info(f"[{session_id}] 🗣️ Language Code: {bulletin_data.language_code}")
        logger.info(f"[{session_id}] 🎙️ Voice Name: {bulletin_data.language_name}")
        logger.info(f"[{session_id}] 🔊 TTS Mode: {'Google Cloud TTS (DYNAMIC)' if USE_GOOGLE_CLOUD_TTS else 'gTTS (fallback)'}")
        logger.info(f"[{session_id}] 📁 Session: {session_id}")
        logger.info(f"{'='*80}\n")
        
        # Log voice details for debugging
        if USE_GOOGLE_CLOUD_TTS:
            logger.info(f"[{session_id}] 🎯 DYNAMIC VOICE DETAILS:")
            logger.info(f"  ✅ Google Cloud TTS is enabled")
            logger.info(f"  🎙️ Requested voice: {bulletin_data.language_name}")
            logger.info(f"  🗣️ Language code: {bulletin_data.language_code}")
            logger.info(f"  📊 This voice will be used for ALL text segments")
        else:
            logger.info(f"[{session_id}] ⚠️ FALLBACK MODE:")
            logger.info(f"  ❌ Google Cloud TTS not available")
            logger.info(f"  🔄 Using gTTS with language: {normalize_voice_to_gtts(bulletin_data.language_name)}")
        
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
        
        logger.info(f"[{session_id}] 📥 Downloading {len(download_tasks)} resources...")
        download_results = await asyncio.gather(*[task[1] for task in download_tasks])
        
        # Map results
        for (key, _), result in zip(download_tasks, download_results):
            download_map[key] = result
        
        background_path = download_map.get("background")
        logo_path = download_map.get("logo")
        frame_path = download_map.get("frame")
        
        # Process segments with DYNAMIC VOICE
        segment_data = []
        total_duration = 0.0
        
        logger.info(f"[{session_id}] 📊 Processing {len(bulletin_data.content)} segments...")
        
        for idx, segment in enumerate(bulletin_data.content):
            logger.info(f"\n[{session_id}] Segment {idx+1}:")
            logger.info(f"  Type: {segment.segment_type}")
            logger.info(f"  Has text: {bool(segment.text)}")
            logger.info(f"  Text length: {len(segment.text) if segment.text else 0} chars")
            logger.info(f"  Will use voice: {bulletin_data.language_name}")
            logger.info(f"  Will use language: {bulletin_data.language_code}")
            
            media_path = download_map.get(f"media_{idx}")
            
            # Duration calculation
            if segment.duration and 0.5 <= segment.duration <= 300:  # Reasonable bounds
                duration = segment.duration
                logger.info(f"  Using provided duration: {duration:.1f}s")
            elif segment.text and segment.text.strip():
                duration = estimate_tts_duration_fixed(segment.text)
                logger.info(f"  Calculated TTS duration: {duration:.1f}s")
            elif media_path and str(media_path).lower().endswith((".mp4", ".avi", ".mov")):
                duration = await asyncio.get_event_loop().run_in_executor(
                    executor, get_video_duration_fast, media_path, session_id
                )
                duration = min(duration, 30.0)  # Cap video duration for performance
                logger.info(f"  Video duration (capped): {duration:.1f}s")
            else:
                duration = 8.0  # Default reasonable duration
                logger.info(f"  Using default duration: {duration:.1f}s")
            
            segment_data.append({
                "segment": segment,
                "media_path": media_path,
                "duration": duration
            })
            total_duration += duration
        
        # SAFETY: Cap total duration to prevent excessive processing
        if total_duration > 600:  # 10 minutes max
            logger.warning(f"[{session_id}] Total duration {total_duration:.1f}s exceeds 10min, scaling down...")
            scale_factor = 600 / total_duration
            for data in segment_data:
                data["duration"] *= scale_factor
            total_duration = 600
        
        logger.info(f"\n[{session_id}] Final total duration: {total_duration:.1f}s")
        
        # =====================================================
        # FIXED BACKGROUND CREATION - ENSURES FULL DURATION
        # =====================================================
        logger.info(f"\n[{session_id}] Building background layer...")
        
        if background_path and background_path.exists():
            try:
                if str(background_path).lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                    logger.info(f"[{session_id}] Processing video background...")
                    bg_clip = VideoFileClip(str(background_path))
                    
                    # Get original background duration
                    bg_duration = bg_clip.duration if bg_clip.duration else 10.0
                    logger.info(f"[{session_id}] Original background duration: {bg_duration:.1f}s")
                    logger.info(f"[{session_id}] Required total duration: {total_duration:.1f}s")
                    
                    # FIXED: Always ensure background covers the FULL duration
                    if bg_duration < total_duration:
                        # Calculate how many loops we need
                        loops_needed = math.ceil(total_duration / bg_duration)
                        logger.info(f"[{session_id}] Background too short, looping {loops_needed} times")
                        
                        # Create looped background
                        bg_list = []
                        remaining_time = total_duration
                        
                        for i in range(loops_needed):
                            if remaining_time <= 0:
                                break
                            
                            if remaining_time >= bg_duration:
                                # Full loop
                                bg_list.append(bg_clip)
                                remaining_time -= bg_duration
                            else:
                                # Partial loop for remaining time
                                partial_clip = bg_clip.subclip(0, remaining_time)
                                bg_list.append(partial_clip)
                                remaining_time = 0
                        
                        # Concatenate all loops
                        bg_clip = concatenate_videoclips(bg_list, method="compose")
                    
                    # CRITICAL: Set exact duration and resize
                    background_clip = bg_clip.resize((VIDEO_WIDTH, VIDEO_HEIGHT))
                    background_clip = background_clip.set_duration(total_duration)
                    background_clip = background_clip.without_audio()  # Remove background audio
                    
                    logger.info(f"[{session_id}] ✅ Background clip created with duration: {background_clip.duration:.1f}s")
                    
                else:
                    # Static image background
                    logger.info(f"[{session_id}] Processing image background...")
                    img = Image.open(background_path).convert("RGB").resize((VIDEO_WIDTH, VIDEO_HEIGHT))
                    background_clip = ImageClip(np.array(img), duration=total_duration)
                    logger.info(f"[{session_id}] ✅ Static background created with duration: {total_duration:.1f}s")
                    
            except Exception as e:
                logger.error(f"[{session_id}] Background error: {e}")
                background_clip = ColorClip(size=(VIDEO_WIDTH, VIDEO_HEIGHT), color=(25, 30, 40), duration=total_duration)
                logger.info(f"[{session_id}] ⚠️ Using fallback background with duration: {total_duration:.1f}s")
        else:
            background_clip = ColorClip(size=(VIDEO_WIDTH, VIDEO_HEIGHT), color=(25, 30, 40), duration=total_duration)
            logger.info(f"[{session_id}] ⚠️ No background found, using solid color with duration: {total_duration:.1f}s")
        
        clips_to_close.append(background_clip)
        
        # Process segments in parallel with DYNAMIC VOICE
        segment_tasks = []
        for data in segment_data:
            seg = data["segment"]
            segment_tasks.append(
                process_video_segment_async(
                    media_path=data["media_path"],
                    text=seg.text,
                    voice_name=bulletin_data.language_name,  # Dynamic voice from JSON
                    language_code=bulletin_data.language_code,  # Dynamic language from JSON
                    segment_type=seg.segment_type,
                    session_dirs=session_dirs,
                    session_id=session_id
                )
            )
        
        logger.info(f"[{session_id}] Processing segments with voice: {bulletin_data.language_name}")
        segment_results = await asyncio.gather(*segment_tasks)
        
        # Build composite with actual durations
        all_clips = [background_clip]  # Background goes first and covers full duration
        current_time = 0.0
        actual_durations = []
        
        for idx, (result, data) in enumerate(zip(segment_results, segment_data)):
            clip, actual_duration = result
            
            if clip:
                clip = clip.set_start(current_time)
                clips_to_close.append(clip)
                all_clips.append(clip)
                actual_durations.append(actual_duration)
                logger.info(f"[{session_id}] Segment {idx+1} actual duration: {actual_duration:.1f}s")
            else:
                actual_durations.append(data["duration"])
            
            current_time += actual_durations[-1]
        
        # Update total duration if segments changed
        final_total_duration = sum(actual_durations)
        
        # CRITICAL: Ensure background duration matches final total
        if abs(final_total_duration - total_duration) > 0.5:  # If there's significant difference
            logger.info(f"[{session_id}] Adjusting background duration: {total_duration:.1f}s -> {final_total_duration:.1f}s")
            background_clip = background_clip.set_duration(final_total_duration)
            total_duration = final_total_duration
        
        logger.info(f"[{session_id}] Final total duration: {total_duration:.1f}s")
        logger.info(f"[{session_id}] Background duration: {background_clip.duration:.1f}s")
        
        # Add overlays
        current_time = 0.0
        
        logger.info(f"[{session_id}] Adding overlays...")
        
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
                
                # Add ticker
                ticker_clip = create_ticker_optimized(bulletin_data.ticker, duration)
                ticker_clip = ticker_clip.set_start(current_time)
                clips_to_close.append(ticker_clip)
                all_clips.append(ticker_clip)
            
            # Add top headline
            if seg.top_headline:
                text_clip = create_text_overlay(seg.top_headline, duration)
                if text_clip:
                    text_clip = text_clip.set_start(current_time)
                    clips_to_close.append(text_clip)
                    all_clips.append(text_clip)
            
            # Add bottom headline
            if seg.bottom_headline and not is_fullscreen:
                bottom_clip = create_bottom_headline(seg.bottom_headline, duration)
                if bottom_clip:
                    bottom_clip = bottom_clip.set_start(current_time)
                    clips_to_close.append(bottom_clip)
                    all_clips.append(bottom_clip)
            
            current_time += duration
        
        # Create final composite
        logger.info(f"\n[{session_id}] Creating final composite...")
        logger.info(f"[{session_id}] Total clips: {len(all_clips)}")
        logger.info(f"[{session_id}] Background duration: {background_clip.duration:.1f}s")
        logger.info(f"[{session_id}] Expected total: {total_duration:.1f}s")
        
        final_video = CompositeVideoClip(all_clips, size=(VIDEO_WIDTH, VIDEO_HEIGHT))
        final_video = final_video.set_duration(total_duration)
        
        logger.info(f"[{session_id}] ✅ Final video duration: {final_video.duration:.1f}s")
        
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
        
        # Render video with optimized settings
        video_filename = f"{unique_id}.mp4"
        video_path = VIDEO_BULLETIN_DIR / video_filename
        
        logger.info(f"\n[{session_id}] 🎬 Rendering video...")
        logger.info(f"  Output: {video_filename}")
        logger.info(f"  Duration: {total_duration:.1f}s")
        logger.info(f"  Language: {bulletin_data.language_code}")
        logger.info(f"  Voice: {bulletin_data.language_name}")
        logger.info(f"  TTS Mode: {'Google Cloud TTS (DYNAMIC)' if USE_GOOGLE_CLOUD_TTS else 'gTTS (fallback)'}")
        
        # OPTIMIZED rendering settings
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
            threads=6,
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
        
        logger.info(f"\n[{session_id}] ✅ SUCCESS! BACKGROUND PERSISTS THROUGHOUT VIDEO")
        logger.info(f"  File: {video_filename}")
        logger.info(f"  Size: {file_size_mb:.1f} MB")
        logger.info(f"  Duration: {total_duration:.1f}s")
        logger.info(f"  Processing time: {processing_time:.1f}s")
        logger.info(f"  Language: {bulletin_data.language_code}")
        logger.info(f"  Voice: {bulletin_data.language_name}")
        logger.info(f"  TTS Mode: {'Google Cloud TTS (DYNAMIC)' if USE_GOOGLE_CLOUD_TTS else 'gTTS fallback'}")
        logger.info(f"  🎬 Background video runs for full {total_duration:.1f} seconds!")
        
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
    """Generate news bulletin with dynamic Google Cloud TTS voice selection"""
    try:
        session_id = SessionManager.create_session_id()
        logger.info(f"\n🚀 NEW BULLETIN REQUEST - Session: {session_id}")
        logger.info(f"🗣️ Language Code: {request.language_code}")
        logger.info(f"🎙️ Voice Name: {request.language_name}")
        logger.info(f"🔊 TTS Mode: {'Google Cloud TTS (DYNAMIC)' if USE_GOOGLE_CLOUD_TTS else 'gTTS (fallback)'}")
        
        # Log request details for debugging
        logger.info(f"📊 Request details:")
        logger.info(f"  Segments: {len(request.content)}")
        logger.info(f"  Ticker: {request.ticker[:50]}..." if len(request.ticker) > 50 else f"  Ticker: {request.ticker}")
        
        video_url = await process_bulletin_optimized(request, session_id)
        
        return {
            "video_url": video_url, 
            "session_id": session_id,
            "voice_used": request.language_name,
            "language_used": request.language_code,
            "tts_mode": "Google Cloud TTS (DYNAMIC)" if USE_GOOGLE_CLOUD_TTS else "gTTS (fallback)"
        }
        
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

@app.get("/")
async def root():
    """API information endpoint with voice examples"""
    return {
        "name": "Dynamic Google Cloud TTS News Bulletin Generator",
        "version": "18.0.0",
        "status": "ready",
        "tts_status": {
            "mode": "Google Cloud TTS (DYNAMIC)" if USE_GOOGLE_CLOUD_TTS else "gTTS (fallback)",
            "client_initialized": tts_client is not None,
            "key_file_found": Path("key.json").exists()
        },
        "features": [
            "✅ DYNAMIC voice selection from JSON request",
            "✅ Google Cloud TTS with key.json authentication",
            "✅ Support for ALL Google Cloud TTS voices",
            "✅ Real-time voice switching per request",
            "✅ Male/Female voice support",
            "✅ Multi-language support",
            "✅ Automatic fallback to gTTS",
            "✅ High-quality neural and WaveNet voices"
        ],
        "voice_examples": {
            "hindi_voices": [
                {
                    "name": "hi-IN-Standard-A",
                    "description": "Hindi Female Standard Voice",
                    "gender": "Female",
                    "type": "Standard"
                },
                {
                    "name": "hi-IN-Standard-B", 
                    "description": "Hindi Male Standard Voice",
                    "gender": "Male",
                    "type": "Standard"
                },
                {
                    "name": "hi-IN-Wavenet-A",
                    "description": "Hindi Female WaveNet Voice (High Quality)",
                    "gender": "Female", 
                    "type": "WaveNet"
                },
                {
                    "name": "hi-IN-Wavenet-B",
                    "description": "Hindi Male WaveNet Voice (High Quality)",
                    "gender": "Male",
                    "type": "WaveNet"
                },
                {
                    "name": "hi-IN-Wavenet-C",
                    "description": "Hindi Female WaveNet Voice (High Quality)",
                    "gender": "Female",
                    "type": "WaveNet"
                },
                {
                    "name": "hi-IN-Wavenet-D",
                    "description": "Hindi Male WaveNet Voice (High Quality)",
                    "gender": "Male",
                    "type": "WaveNet"
                },
                {
                    "name": "hi-IN-Neural2-A",
                    "description": "Hindi Female Neural2 Voice (Premium)",
                    "gender": "Female",
                    "type": "Neural2"
                },
                {
                    "name": "hi-IN-Neural2-B",
                    "description": "Hindi Male Neural2 Voice (Premium)",
                    "gender": "Male", 
                    "type": "Neural2"
                },
                {
                    "name": "hi-IN-Neural2-C",
                    "description": "Hindi Male Neural2 Voice (Premium)",
                    "gender": "Male",
                    "type": "Neural2"
                },
                {
                    "name": "hi-IN-Neural2-D",
                    "description": "Hindi Female Neural2 Voice (Premium)",
                    "gender": "Female",
                    "type": "Neural2"
                }
            ],
            "english_india_voices": [
                {
                    "name": "en-IN-Standard-A",
                    "description": "English (India) Female Standard Voice",
                    "gender": "Female",
                    "type": "Standard"
                },
                {
                    "name": "en-IN-Standard-B",
                    "description": "English (India) Male Standard Voice", 
                    "gender": "Male",
                    "type": "Standard"
                },
                {
                    "name": "en-IN-Wavenet-A",
                    "description": "English (India) Female WaveNet Voice",
                    "gender": "Female",
                    "type": "WaveNet"
                },
                {
                    "name": "en-IN-Wavenet-B",
                    "description": "English (India) Male WaveNet Voice",
                    "gender": "Male",
                    "type": "WaveNet"
                },
                {
                    "name": "en-IN-Wavenet-C",
                    "description": "English (India) Male WaveNet Voice",
                    "gender": "Male",
                    "type": "WaveNet"
                },
                {
                    "name": "en-IN-Wavenet-D",
                    "description": "English (India) Female WaveNet Voice",
                    "gender": "Female",
                    "type": "WaveNet"
                }
            ],
            "other_indian_languages": [
                {
                    "name": "bn-IN-Standard-A",
                    "description": "Bengali (India) Female Standard Voice",
                    "gender": "Female",
                    "language": "Bengali"
                },
                {
                    "name": "bn-IN-Standard-B", 
                    "description": "Bengali (India) Male Standard Voice",
                    "gender": "Male",
                    "language": "Bengali"
                },
                {
                    "name": "bn-IN-Wavenet-A",
                    "description": "Bengali (India) Female WaveNet Voice",
                    "gender": "Female",
                    "language": "Bengali"
                },
                {
                    "name": "bn-IN-Wavenet-B",
                    "description": "Bengali (India) Male WaveNet Voice",
                    "gender": "Male",
                    "language": "Bengali"
                },
                {
                    "name": "ta-IN-Standard-A",
                    "description": "Tamil (India) Female Standard Voice",
                    "gender": "Female", 
                    "language": "Tamil"
                },
                {
                    "name": "ta-IN-Standard-B",
                    "description": "Tamil (India) Male Standard Voice",
                    "gender": "Male",
                    "language": "Tamil"
                },
                {
                    "name": "ta-IN-Wavenet-A",
                    "description": "Tamil (India) Female WaveNet Voice",
                    "gender": "Female",
                    "language": "Tamil"
                },
                {
                    "name": "ta-IN-Wavenet-B",
                    "description": "Tamil (India) Male WaveNet Voice", 
                    "gender": "Male",
                    "language": "Tamil"
                },
                {
                    "name": "te-IN-Standard-A",
                    "description": "Telugu (India) Female Standard Voice",
                    "gender": "Female",
                    "language": "Telugu"
                },
                {
                    "name": "te-IN-Standard-B",
                    "description": "Telugu (India) Male Standard Voice",
                    "gender": "Male",
                    "language": "Telugu"
                },
                {
                    "name": "mr-IN-Standard-A",
                    "description": "Marathi (India) Female Standard Voice",
                    "gender": "Female",
                    "language": "Marathi"
                },
                {
                    "name": "mr-IN-Standard-B",
                    "description": "Marathi (India) Male Standard Voice",
                    "gender": "Male", 
                    "language": "Marathi"
                },
                {
                    "name": "mr-IN-Wavenet-A",
                    "description": "Marathi (India) Female WaveNet Voice",
                    "gender": "Female",
                    "language": "Marathi"
                },
                {
                    "name": "mr-IN-Wavenet-B", 
                    "description": "Marathi (India) Male WaveNet Voice",
                    "gender": "Male",
                    "language": "Marathi"
                },
                {
                    "name": "gu-IN-Standard-A",
                    "description": "Gujarati (India) Female Standard Voice",
                    "gender": "Female",
                    "language": "Gujarati"
                },
                {
                    "name": "gu-IN-Standard-B",
                    "description": "Gujarati (India) Male Standard Voice", 
                    "gender": "Male",
                    "language": "Gujarati"
                },
                {
                    "name": "gu-IN-Wavenet-A",
                    "description": "Gujarati (India) Female WaveNet Voice",
                    "gender": "Female",
                    "language": "Gujarati"
                },
                {
                    "name": "gu-IN-Wavenet-B",
                    "description": "Gujarati (India) Male WaveNet Voice",
                    "gender": "Male",
                    "language": "Gujarati"
                },
                {
                    "name": "kn-IN-Standard-A",
                    "description": "Kannada (India) Female Standard Voice", 
                    "gender": "Female",
                    "language": "Kannada"
                },
                {
                    "name": "kn-IN-Standard-B",
                    "description": "Kannada (India) Male Standard Voice",
                    "gender": "Male",
                    "language": "Kannada"
                },
                {
                    "name": "kn-IN-Wavenet-A",
                    "description": "Kannada (India) Female WaveNet Voice",
                    "gender": "Female",
                    "language": "Kannada"
                },
                {
                    "name": "kn-IN-Wavenet-B",
                    "description": "Kannada (India) Male WaveNet Voice",
                    "gender": "Male",
                    "language": "Kannada"
                },
                {
                    "name": "ml-IN-Standard-A",
                    "description": "Malayalam (India) Female Standard Voice",
                    "gender": "Female",
                    "language": "Malayalam"
                },
                {
                    "name": "ml-IN-Wavenet-A",
                    "description": "Malayalam (India) Female WaveNet Voice",
                    "gender": "Female", 
                    "language": "Malayalam"
                },
                {
                    "name": "ml-IN-Wavenet-B",
                    "description": "Malayalam (India) Male WaveNet Voice",
                    "gender": "Male",
                    "language": "Malayalam"
                }
            ]
        },
        "setup_instructions": {
            "required_file": "key.json",
            "description": "Place your Google Cloud service account key as 'key.json' in the app directory",
            "steps": [
                "1. Create a Google Cloud Project",
                "2. Enable the Text-to-Speech API",
                "3. Create a service account with TTS permissions",
                "4. Download the JSON credentials file",
                "5. Rename it to 'key.json' and place in app directory",
                "6. Restart the application"
            ]
        },
        "usage_example": {
            "description": "Send a POST request to /generate-bulletin with dynamic voice selection",
            "sample_json": {
                "logo_url": "https://example.com/logo.png",
                "language_code": "hi-IN",
                "language_name": "hi-IN-Wavenet-D",
                "ticker": "Your breaking news ticker text...",
                "background_url": "https://example.com/background.mp4",
                "content": [
                    {
                        "segment_type": "main_news",
                        "media_url": "https://example.com/news-image.jpg",
                        "text": "Your news content in Hindi...",
                        "top_headline": "Top Headline",
                        "bottom_headline": "Bottom Headline"
                    }
                ]
            },
            "note": "The voice specified in 'language_name' will be used for ALL text segments in the bulletin"
        }
    }

@app.get("/status")
async def get_status():
    """System status with detailed TTS info"""
    try:
        video_count = len(list(VIDEO_BULLETIN_DIR.glob("*.mp4")))
        thumb_count = len(list(THUMBNAILS_DIR.glob("*.jpg")))
        temp_sessions = len(list(TEMP_BASE_DIR.iterdir()))
        key_file_exists = Path("key.json").exists()
        
        return {
            "status": "operational",
            "version": "18.0.0 - Dynamic Google Cloud TTS Edition", 
            "tts_status": {
                "mode": "Google Cloud TTS (DYNAMIC)" if USE_GOOGLE_CLOUD_TTS else "gTTS (fallback)",
                "google_cloud_enabled": USE_GOOGLE_CLOUD_TTS,
                "client_initialized": tts_client is not None,
                "key_file_found": key_file_exists,
                "key_file_location": str(Path("key.json").absolute()) if key_file_exists else "Not found",
                "dynamic_voice_support": USE_GOOGLE_CLOUD_TTS
            },
            "statistics": {
                "videos_generated": video_count,
                "thumbnails": thumb_count,
                "active_sessions": temp_sessions,
                "cache_items": len(REQUEST_CACHE)
            },
            "features": [
                f"Google Cloud TTS with dynamic voice selection" if USE_GOOGLE_CLOUD_TTS else "gTTS fallback mode",
                "Real-time voice switching per request",
                "Support for ALL Google Cloud TTS voices",
                "Male/Female voice selection",
                "Multi-language support",
                "High-quality Neural2 and WaveNet voices",
                "Duration calculation optimized",
                "Smart video looping",
                "Optimized rendering pipeline"
            ],
            "performance": {
                "max_workers": MAX_WORKERS,
                "render_preset": RENDER_PRESET,
                "render_fps": RENDER_FPS,
                "max_duration": "10 minutes"
            },
            "voice_capabilities": {
                "total_supported_voices": "200+ Google Cloud TTS voices" if USE_GOOGLE_CLOUD_TTS else "60+ gTTS languages",
                "quality_levels": ["Standard", "WaveNet", "Neural2"] if USE_GOOGLE_CLOUD_TTS else ["Standard"],
                "languages_supported": ["Hindi", "English", "Bengali", "Tamil", "Telugu", "Marathi", "Gujarati", "Kannada", "Malayalam", "And many more..."]
            }
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
        
        for path in [VIDEO_BULLETIN_DIR, THUMBNAILS_DIR]:
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

# ==============================================================================
# STARTUP & SHUTDOWN
# ==============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("\n" + "="*80)
    logger.info(" " * 5 + "NEWS BULLETIN GENERATOR v18.0 - DYNAMIC GOOGLE CLOUD TTS")
    logger.info("="*80)
    
    # Check for key.json file first
    key_path = Path("key.json")
    if key_path.exists():
        logger.info(f"\n📁 FOUND: {key_path.absolute()}")
        logger.info(f"📊 File size: {key_path.stat().st_size} bytes")
    else:
        logger.info(f"\n❌ NOT FOUND: {key_path.absolute()}")
        logger.info("💡 Place your Google Cloud service account key as 'key.json' in the app directory")
    
    # Initialize Google Cloud TTS
    tts_success = initialize_google_cloud_tts()
    
    if tts_success:
        logger.info("\n✅ GOOGLE CLOUD TTS INITIALIZED SUCCESSFULLY!")
        logger.info("🎙️ DYNAMIC VOICE SUPPORT ENABLED!")
        logger.info("📍 Features available:")
        logger.info("  • Real-time voice selection from JSON requests")
        logger.info("  • Support for ALL Google Cloud TTS voices")  
        logger.info("  • Male/Female voice switching")
        logger.info("  • Standard, WaveNet, and Neural2 voice quality")
        logger.info("  • Multi-language support")
        logger.info("\n🎯 HOW IT WORKS:")
        logger.info("  1. Send 'language_name' in your JSON request")
        logger.info("  2. Voice is dynamically selected for that request")
        logger.info("  3. All text segments use the specified voice")
        logger.info("  4. Different requests can use different voices!")
        
        logger.info("\n🗣️ VOICE EXAMPLES:")
        logger.info("  Hindi voices:")
        logger.info('    • "hi-IN-Standard-A" (Female)')
        logger.info('    • "hi-IN-Wavenet-D" (Male, High Quality)')
        logger.info('    • "hi-IN-Neural2-B" (Male, Premium)')
        logger.info("  English (India) voices:")
        logger.info('    • "en-IN-Wavenet-A" (Female)')
        logger.info('    • "en-IN-Wavenet-B" (Male)')
        logger.info("  Other languages:")
        logger.info('    • "bn-IN-Wavenet-A" (Bengali Female)')
        logger.info('    • "ta-IN-Wavenet-B" (Tamil Male)')
        logger.info('    • "te-IN-Standard-A" (Telugu Female)')
        logger.info('    • "mr-IN-Wavenet-A" (Marathi Female)')
        logger.info("    • And 200+ more voices!")
        
    else:
        logger.info("\n⚠️ GOOGLE CLOUD TTS NOT CONFIGURED")
        logger.info("📍 Using gTTS fallback (basic language support only)")
        logger.info("💡 TO ENABLE DYNAMIC VOICE SUPPORT:")
        logger.info("  1. Create a Google Cloud project")
        logger.info("  2. Enable Text-to-Speech API")
        logger.info("  3. Create service account credentials")
        logger.info("  4. Download the JSON key file")
        logger.info("  5. Rename to 'key.json' and place in app directory")
        logger.info("  6. Restart the application")
    
    logger.info("\n🔧 SYSTEM FEATURES:")
    logger.info("  • Dynamic voice selection per request")
    logger.info("  • Voice switching without restart")
    logger.info("  • High-quality audio generation")
    logger.info("  • Multi-language support") 
    logger.info("  • Smart duration calculation")
    logger.info("  • Optimized video rendering")
    logger.info("  • Automatic cleanup")
    
    logger.info("\n📊 PERFORMANCE SETTINGS:")
    logger.info(f"  • Render preset: {RENDER_PRESET}")
    logger.info(f"  • Workers: {MAX_WORKERS}")
    logger.info(f"  • FPS: {RENDER_FPS}")
    logger.info(f"  • CRF: {RENDER_CRF}")
    
    # Setup fonts
    await setup_fonts_async()
    
    logger.info("\n✅ System ready for dynamic voice bulletin generation!")

@app.on_event("shutdown") 
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("\n🛑 Shutting down...")
    
    # Close HTTP client
    await http_client.aclose()
    
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
    print(" " * 8 + "NEWS BULLETIN GENERATOR v18.0 - DYNAMIC GOOGLE CLOUD TTS")
    print("="*80)
    
    print("\n🎙️ DYNAMIC VOICE SELECTION SYSTEM:")
    
    print("\n  ✅ WHAT'S NEW IN v18.0:")
    print("     • DYNAMIC voice selection from JSON requests")
    print("     • Voice changes per request (no restart needed)")
    print("     • Uses key.json for Google Cloud authentication") 
    print("     • Support for ALL Google Cloud TTS voices")
    print("     • Real-time voice switching")
    print("     • Enhanced voice quality options")
    
    print("\n  📁 SETUP REQUIREMENTS:")
    print("     1. Place your Google Cloud service account key as 'key.json'")
    print("     2. Make sure key.json is in the same directory as this script")
    print("     3. Restart the application")
    print("     4. Send requests with 'language_name' parameter")
    
    print("\n  🎯 HOW TO USE:")
    print("     • Send POST request to /generate-bulletin")
    print("     • Include 'language_name' with specific voice:")
    print('       "language_name": "hi-IN-Wavenet-D"')
    print("     • Each request can use a different voice!")
    print("     • No need to restart the server")
    
    print("\n  📊 YOUR POSTMAN EXAMPLE WILL WORK:")
    print("     {")
    print('       "logo_url": "...",')
    print('       "language_code": "hi-IN",')
    print('       "language_name": "hi-IN-Chirp3-HD-Achird",  # This voice will be used!')
    print('       "ticker": "...",')
    print('       "background_url": "...",')
    print('       "content": [...]')
    print("     }")
    
    print("\n  🗣️ VOICE EXAMPLES:")
    print("     Hindi voices:")
    print('       • "hi-IN-Standard-A" (Female Standard)')
    print('       • "hi-IN-Standard-B" (Male Standard)')
    print('       • "hi-IN-Wavenet-A" (Female High Quality)')
    print('       • "hi-IN-Wavenet-D" (Male High Quality)')
    print('       • "hi-IN-Neural2-A" (Female Premium)')
    print('       • "hi-IN-Neural2-B" (Male Premium)')
    
    print("\n     English (India) voices:")
    print('       • "en-IN-Standard-A" (Female)')
    print('       • "en-IN-Wavenet-B" (Male)')
    print('       • "en-IN-Neural2-A" (Female Premium)')
    
    print("\n     Other languages:")
    print('       • "bn-IN-Wavenet-A" (Bengali Female)')
    print('       • "ta-IN-Wavenet-B" (Tamil Male)')
    print('       • "te-IN-Standard-A" (Telugu Female)')
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")