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

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

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
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install: pip install fastapi uvicorn opencv-python-headless pillow numpy moviepy requests gtts")
    raise SystemExit(1)

# --------------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("bulletin")

# --------------------------------------------------------------------------------------
# FastAPI app
# --------------------------------------------------------------------------------------
app = FastAPI(title="News Bulletin Generator API", version="9.0.2")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------------------
# Directories
# --------------------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
THUMBNAILS_DIR = BASE_DIR / "thumbnails"
VIDEO_BULLETIN_DIR = BASE_DIR / "video-bulletin"
TEMP_DIR = BASE_DIR / "temp"
FONTS_DIR = BASE_DIR / "fonts"
AUDIO_DIR = BASE_DIR / "audio"

for p in [THUMBNAILS_DIR, VIDEO_BULLETIN_DIR, TEMP_DIR, FONTS_DIR, AUDIO_DIR]:
    p.mkdir(parents=True, exist_ok=True)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key.json"

# --------------------------------------------------------------------------------------
# Layout constants
# --------------------------------------------------------------------------------------
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080

CONTENT_X = 40
CONTENT_Y = 40
CONTENT_WIDTH = 1840
CONTENT_HEIGHT = 900

TICKER_HEIGHT = 120
TICKER_Y = VIDEO_HEIGHT - TICKER_HEIGHT

LOGO_SIZE = 150
LOGO_X = VIDEO_WIDTH - LOGO_SIZE - 80
LOGO_Y = 30

# ADJUSTED: Move text overlay higher up so it fits within frame
TEXT_OVERLAY_X = 60
TEXT_OVERLAY_Y = 40  # Changed from 50 to 40 - moved up
TEXT_OVERLAY_WIDTH = 800
TEXT_OVERLAY_HEIGHT = 100

# --------------------------------------------------------------------------------------
# Fonts
# --------------------------------------------------------------------------------------
HINDI_FONT_URL = "https://github.com/google/fonts/raw/main/ofl/notosansdevanagari/NotoSansDevanagari%5Bwdth%2Cwght%5D.ttf"
HINDI_FONT_PATH = FONTS_DIR / "hindi-bold.ttf"
BOLD_FONT_URL = "https://github.com/google/fonts/raw/main/apache/robotocondensed/RobotoCondensed-Bold.ttf"
BOLD_FONT_PATH = FONTS_DIR / "roboto-bold.ttf"

def _download_font_once(url: str, path: Path) -> None:
    try:
        if path.exists():
            return
        logger.info(f"Downloading font: {url}")
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
        logger.info(f"Saved font to {path}")
    except Exception as e:
        logger.warning(f"Font download warning for {url}: {e}")

def setup_fonts() -> None:
    _download_font_once(HINDI_FONT_URL, HINDI_FONT_PATH)
    _download_font_once(BOLD_FONT_URL, BOLD_FONT_PATH)

setup_fonts()

def _load_font(size: int) -> ImageFont.FreeTypeFont:
    try:
        if HINDI_FONT_PATH.exists():
            return ImageFont.truetype(str(HINDI_FONT_PATH), size=size, encoding="utf-8")
        if BOLD_FONT_PATH.exists():
            return ImageFont.truetype(str(BOLD_FONT_PATH), size=size)
    except Exception as e:
        logger.warning(f"Loading truetype font failed: {e}")
    return ImageFont.load_default()

# --------------------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------------------
class ContentSegment(BaseModel):
    segment_type: str
    media_url: Optional[str] = None
    frame_url: Optional[str] = None
    text: Optional[str] = None
    top_headline: Optional[str] = None
    duration: Optional[float] = None

class BulletinData(BaseModel):
    id: str
    logo_url: str
    language_code: str = "hi-IN"
    language_name: str = "Hindi"
    ticker: str
    background_url: str
    story_thumbnail: Optional[str] = None
    generated_story_url: Optional[str] = None
    content: List[ContentSegment]

class BulletinRequest(BaseModel):
    data: List[BulletinData]

class BulletinResponse(BaseModel):
    status: bool
    message: str
    data: List[Dict[str, Any]]

# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------
def _normalize_gtts_lang(code: str) -> str:
    """
    gTTS supported languages are mostly 2-letter (e.g., 'hi', 'en').
    Normalize 'hi-IN' -> 'hi', 'en-US' -> 'en', etc.
    """
    if not code:
        return "hi"
    lc = code.lower().replace("_", "-")
    if lc.startswith("hi"):
        return "hi"
    if "-" in lc:
        return lc.split("-")[0]
    return lc

def _safe_unlink(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except:
        pass

def _requests_get(url: str, **kwargs):
    """
    Simple retry wrapper for GET requests.
    """
    last_err = None
    for _ in range(2):
        try:
            return requests.get(url, **kwargs)
        except Exception as e:
            last_err = e
            time.sleep(0.3)
    if last_err:
        raise last_err

def download_file(url: str, destination: Path = TEMP_DIR) -> Optional[Path]:
    """
    Download file (video/image/base64-data) to TEMP_DIR.
    Returns local Path or None on failure.
    """
    try:
        if not url:
            logger.warning("No URL provided for download")
            return None

        # Base64 data URL
        if url.startswith("data:"):
            header, data = url.split(",", 1)
            file_data = base64.b64decode(data)
            ext = ".png" if "png" in header else ".jpg"
            file_path = destination / f"{uuid.uuid4()}{ext}"
            with open(file_path, "wb") as f:
                f.write(file_data)
            if file_path.exists() and file_path.stat().st_size > 0:
                logger.info(f"Downloaded base64 to {file_path}")
                return file_path
            logger.error("Base64 file empty")
            return None

        r = _requests_get(url, stream=True, timeout=30)
        r.raise_for_status()
        content_type = r.headers.get("content-type", "")

        if "video" in content_type:
            ext = ".mp4"
        elif "png" in content_type:
            ext = ".png"
        elif "jpeg" in content_type or "jpg" in content_type:
            ext = ".jpg"
        else:
            # fallback by URL suffix
            ext = Path(url.split("?")[0]).suffix or ".mp4"

        file_path = destination / f"{uuid.uuid4()}{ext}"
        with open(file_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        if file_path.exists() and file_path.stat().st_size > 0:
            logger.info(f"Downloaded: {file_path.name} ({file_path.stat().st_size/1024:.0f} KB)")
            return file_path

        logger.error(f"Empty download: {url}")
        return None
    except Exception as e:
        logger.error(f"Download failed for {url}: {e}")
        return None

def _make_silence(duration: float, fps: int = 44100, nch: int = 2) -> AudioArrayClip:
    if duration <= 0:
        duration = 0.05
    nframes = int(round(duration * fps))
    if nframes < 1:
        nframes = 1
    silent_arr = np.zeros((nframes, nch), dtype=np.float32)
    return AudioArrayClip(silent_arr, fps=fps)

def _make_gtts_audio(text: str, lang_code: str, target_duration: float) -> Optional[AudioFileClip]:
    """
    Create TTS audio with proper duration handling
    """
    if not text or not text.strip():
        logger.warning("No text provided for TTS")
        return None
    
    try:
        lang = _normalize_gtts_lang(lang_code)
        audio_path = AUDIO_DIR / f"audio_{uuid.uuid4()}.mp3"
        
        logger.info(f"Creating TTS audio: lang={lang}, text_length={len(text)}")
        
        # Try to create TTS with retry
        last_err = None
        for attempt in range(3):
            try:
                tts = gTTS(text=text, lang=lang, slow=False)
                tts.save(str(audio_path))
                if audio_path.exists() and audio_path.stat().st_size > 0:
                    logger.info(f"TTS file created: {audio_path.name} ({audio_path.stat().st_size} bytes)")
                    break
            except Exception as e:
                last_err = e
                logger.warning(f"TTS attempt {attempt+1} failed: {e}")
                time.sleep(0.5)
        
        if not audio_path.exists() or audio_path.stat().st_size == 0:
            logger.error(f"TTS file creation failed: {last_err}")
            return None
        
        # Load the TTS audio
        tts_clip = AudioFileClip(str(audio_path))
        logger.info(f"TTS audio loaded: duration={tts_clip.duration:.2f}s")
        
        # Speed adjustment if needed
        if tts_clip.duration > target_duration * 1.2 and tts_clip.duration > 0:
            speed_factor = tts_clip.duration / target_duration
            logger.info(f"Speeding up audio by {speed_factor:.2f}x to fit duration")
            tts_clip = tts_clip.fx(vfx.speedx, speed_factor)
        
        # Create a base silence track with target duration
        final_duration = max(target_duration, tts_clip.duration) + 0.1
        silence = _make_silence(final_duration, fps=44100, nch=2)
        
        # Composite the TTS on top of silence
        final_audio = CompositeAudioClip([silence, tts_clip.set_start(0)])
        final_audio = final_audio.set_duration(final_duration)
        
        logger.info(f"Final TTS audio: duration={final_audio.duration:.2f}s")
        return final_audio
        
    except Exception as e:
        logger.error(f"TTS audio creation error: {e}", exc_info=True)
        return None

def check_video_has_audio(video_path: Path) -> bool:
    try:
        v = VideoFileClip(str(video_path))
        has_audio = v.audio is not None
        if has_audio:
            try:
                arr = v.audio.to_soundarray(fps=22050)
                has_audio = np.any(np.abs(arr) > 1e-4)
            except Exception:
                pass
        v.close()
        return has_audio
    except Exception as e:
        logger.error(f"check_video_has_audio error: {e}")
        return False

# --------------------------------------------------------------------------------------
# Visual layers
# --------------------------------------------------------------------------------------
def create_logo_overlay(duration: float, logo_path: Optional[Path]) -> ImageClip:
    """
    Render a circular logo with a subtle white plate behind.
    """
    try:
        overlay = Image.new("RGBA", (VIDEO_WIDTH, VIDEO_HEIGHT), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # circular plate
        plate_size = LOGO_SIZE + 20
        draw.ellipse([LOGO_X - 10, LOGO_Y - 10,
                      LOGO_X + plate_size - 10, LOGO_Y + plate_size - 10],
                     fill=(255, 255, 255, 230),
                     outline=(220, 20, 60, 255), width=3)

        if logo_path and logo_path.exists():
            try:
                logo = Image.open(logo_path).convert("RGBA")
                ratio = min(LOGO_SIZE / logo.width, LOGO_SIZE / logo.height)
                new_w = int(logo.width * ratio * 0.9)
                new_h = int(logo.height * ratio * 0.9)
                logo = logo.resize((new_w, new_h), Image.Resampling.LANCZOS)
                lx = LOGO_X + (LOGO_SIZE - new_w) // 2
                ly = LOGO_Y + (LOGO_SIZE - new_h) // 2
                overlay.paste(logo, (lx, ly), logo)
            except Exception as e:
                logger.error(f"Logo overlay error: {e}")

        return ImageClip(np.array(overlay), transparent=True, duration=duration)
    except Exception as e:
        logger.error(f"create_logo_overlay error: {e}")
        return ImageClip(np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 4), dtype=np.uint8),
                         transparent=True, duration=duration)

def create_frame_border(frame_path: Optional[Path], duration: float) -> ImageClip:
    """
    If frame image provided, stretch to full screen; else draw a simple border around content area.
    """
    try:
        if frame_path and frame_path.exists():
            img = Image.open(frame_path).convert("RGBA")
            img = img.resize((VIDEO_WIDTH, VIDEO_HEIGHT), Image.Resampling.LANCZOS)
        else:
            img = Image.new("RGBA", (VIDEO_WIDTH, VIDEO_HEIGHT), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            draw.rectangle([CONTENT_X - 10, CONTENT_Y - 10,
                            CONTENT_X + CONTENT_WIDTH + 10, CONTENT_Y + CONTENT_HEIGHT + 10],
                           outline=(255, 215, 0, 255), width=10)
            draw.rectangle([CONTENT_X - 5, CONTENT_Y - 5,
                            CONTENT_X + CONTENT_WIDTH + 5, CONTENT_Y + CONTENT_HEIGHT + 5],
                           outline=(255, 255, 255, 220), width=3)
        return ImageClip(np.array(img), transparent=True, duration=duration)
    except Exception as e:
        logger.error(f"create_frame_border error: {e}")
        return ImageClip(np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 4), dtype=np.uint8),
                         transparent=True, duration=duration)

def _render_text_with_outline(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str, font: ImageFont.FreeTypeFont,
                              fill=(255, 255, 255, 255)) -> None:
    x, y = xy
    # black outline
    for ox, oy in [(-2,0),(2,0),(0,-2),(0,2),(-2,-2),(2,-2),(-2,2),(2,2)]:
        draw.text((x+ox, y+oy), text, font=font, fill=(0, 0, 0, 255))
    draw.text((x, y), text, font=font, fill=fill)

def create_text_overlay_pil(text: str, duration: float) -> Optional[ImageClip]:
    """
    Pure PIL text overlay - BIGGER text size to cover more frame area
    """
    try:
        if not text or not text.strip():
            return None
        canvas = Image.new("RGBA", (VIDEO_WIDTH, VIDEO_HEIGHT), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        # INCREASED font size from 50 to 80 for bigger headline
        font = _load_font(80)
        
        # Position at top left, slightly adjusted for bigger text
        _render_text_with_outline(draw, (TEXT_OVERLAY_X, TEXT_OVERLAY_Y), text, font)
        
        return ImageClip(np.array(canvas), transparent=True, duration=duration)
    except Exception as e:
        logger.error(f"create_text_overlay_pil error: {e}")
        return None

def _measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> Tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    return w, h

def create_scrolling_ticker_pil(ticker_text: str, duration: float, speed_px_per_s: int = 150) -> VideoClip:
    """
    Build a PIL-based scrolling ticker with improved Breaking News badge
    """
    # build base bar
    bar_w, bar_h = VIDEO_WIDTH, TICKER_HEIGHT
    font = _load_font(56)

    text_for_strip = f"  ●  {ticker_text}  ●  {ticker_text}  ●  {ticker_text}  ●  "
    # Pre-render text strip to a long image
    dummy = Image.new("RGB", (10, 10), (0, 0, 0))
    d = ImageDraw.Draw(dummy)
    text_w, text_h = _measure_text(d, text_for_strip, font)
    strip_w = max(text_w + VIDEO_WIDTH, VIDEO_WIDTH * 2)
    strip_h = max(text_h + 10, bar_h - 20)

    strip_img = Image.new("RGBA", (strip_w, strip_h), (0, 0, 0, 0))
    sd = ImageDraw.Draw(strip_img)
    _render_text_with_outline(sd, (0, (strip_h - text_h)//2), text_for_strip, font)

    strip_np = np.array(strip_img)
    
    # BIGGER Breaking News badge dimensions
    badge_w, badge_h = 250, TICKER_HEIGHT - 10  # Made wider and full ticker height

    def make_frame(t: float):
        # base bar with darker red for better contrast
        frame = Image.new("RGB", (bar_w, bar_h), (160, 20, 30))
        
        # gradient shine top (subtle)
        g = ImageDraw.Draw(frame)
        for i in range(bar_h // 3):
            alpha = int(80 - i * 2)
            if alpha < 0: alpha = 0
            g.line([(0, i), (bar_w, i)], fill=(200, 40, 50))

        # compute x based on time - START FROM badge width so text doesn't overlap initially
        shift = int((t * speed_px_per_s) % (strip_w + bar_w))
        x = badge_w + 20 - shift  # Start after badge with some padding
        
        # paste strip on frame (BEFORE badge so it goes behind)
        frame_np = np.array(frame).astype(np.uint8)
        
        # handle partially visible region
        y = (bar_h - strip_h) // 2
        left = max(0, x)
        right = min(bar_w, x + strip_w)
        if right > left:
            src_x1 = left - x
            src_x2 = src_x1 + (right - left)
            dst_x1 = left
            dst_x2 = right
            sub = strip_np[:, src_x1:src_x2, :]
            # alpha composite
            alpha = sub[:, :, 3:4] / 255.0
            frame_np[y:y+strip_h, dst_x1:dst_x2, :] = (
                (1 - alpha) * frame_np[y:y+strip_h, dst_x1:dst_x2, :] + alpha * sub[:, :, :3]
            ).astype(np.uint8)

        # IMPROVED Breaking News badge (drawn AFTER text so it covers it)
        # Full height badge with better design
        badge = Image.new("RGBA", (badge_w, badge_h), (0, 0, 0, 0))
        bd = ImageDraw.Draw(badge)
        
        # Main badge background with gradient effect
        bd.rectangle([0, 0, badge_w, badge_h], fill=(255, 255, 255, 255))
        
        # Red accent bars - top and bottom
        bd.rectangle([0, 0, badge_w, 12], fill=(220, 20, 60))
        bd.rectangle([0, badge_h-12, badge_w, badge_h], fill=(220, 20, 60))
        
        # Side accent
        bd.rectangle([0, 0, 5, badge_h], fill=(220, 20, 60))
        bd.rectangle([badge_w-5, 0, badge_w, badge_h], fill=(220, 20, 60))
        
        # Text with better positioning
        f1 = _load_font(40)  # Bigger font for BREAKING
        f2 = _load_font(32)  # Bigger font for NEWS
        
        # Center the text properly
        breaking_text = "BREAKING"
        news_text = "NEWS"
        
        # Calculate text positions for centering
        b_bbox = bd.textbbox((0, 0), breaking_text, font=f1)
        b_width = b_bbox[2] - b_bbox[0]
        n_bbox = bd.textbbox((0, 0), news_text, font=f2)
        n_width = n_bbox[2] - n_bbox[0]
        
        # Draw centered text
        _render_text_with_outline(bd, ((badge_w - b_width)//2, 20), breaking_text, f1, (220, 20, 60, 255))
        _render_text_with_outline(bd, ((badge_w - n_width)//2, 55), news_text, f2, (40, 40, 40, 255))
        
        badge_np = np.array(badge)

        # paste badge at left corner (covering any text that might be there)
        bx = 0  # Start from absolute left
        by = 5  # Small margin from top
        
        # Direct paste for opaque badge (no alpha blending needed for full opacity areas)
        frame_np[by:by+badge_h, bx:bx+badge_w, :] = badge_np[:, :, :3]

        return frame_np

    clip = VideoClip(make_frame, duration=duration)
    # Place it at bottom
    return clip.set_position((0, TICKER_Y)).set_duration(duration)

# --------------------------------------------------------------------------------------
# Segment processor
# --------------------------------------------------------------------------------------
def process_video_segment(
    media_path: Optional[Path],
    duration: float,
    text: Optional[str] = None,
    language_code: str = "hi",
    loop: bool = True
) -> Optional[VideoFileClip]:
    """
    Returns a clip sized to CONTENT rect and positioned accordingly.
    FIXED: Better audio handling for TTS
    """
    try:
        if media_path and media_path.exists():
            suffix = str(media_path).lower()
            if suffix.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                # Video branch
                v = VideoFileClip(str(media_path))
                if not v:
                    logger.error("VideoClip creation failed")
                    return None

                has_audio = v.audio is not None
                if has_audio:
                    try:
                        arr = v.audio.to_soundarray(fps=22050)
                        has_audio = np.any(np.abs(arr) > 1e-4)
                    except Exception:
                        pass

                # loop to match duration if needed
                if loop and v.duration < duration and v.duration > 0:
                    loops_needed = math.ceil(duration / v.duration)
                    v = concatenate_videoclips([v] * loops_needed, method="compose")

                v = v.subclip(0, duration)
                v = v.resize((CONTENT_WIDTH, CONTENT_HEIGHT)).set_position((CONTENT_X, CONTENT_Y))

                if not has_audio and text and text.strip():
                    logger.info(f"  Video has no audio, adding TTS for: {text[:50]}...")
                    audio = _make_gtts_audio(text, language_code, duration)
                    if audio:
                        v = v.set_audio(audio)
                        logger.info("  ✓ TTS audio successfully added to video")
                    else:
                        logger.error("  ✗ TTS audio creation failed for video")
                else:
                    if has_audio:
                        logger.info("  ✓ Preserving original video audio")
                    else:
                        logger.info("  Video has no audio and no text provided")
                return v

            else:
                # Image branch
                img = Image.open(media_path).convert("RGBA")
                img = img.resize((CONTENT_WIDTH, CONTENT_HEIGHT), Image.Resampling.LANCZOS)
                img_arr = np.array(img)
                if img_arr.shape[2] == 4:
                    # convert to RGB for ImageClip base
                    img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGBA2RGB)
                clip = ImageClip(img_arr, duration=duration).set_position((CONTENT_X, CONTENT_Y))
                # gentle zoom
                clip = clip.resize(lambda t: 1 + 0.02 * t)
                
                if text and text.strip():
                    logger.info(f"  Image segment, adding TTS for: {text[:50]}...")
                    audio = _make_gtts_audio(text, language_code, duration)
                    if audio:
                        clip = clip.set_audio(audio)
                        logger.info("  ✓ TTS audio successfully added to image")
                    else:
                        logger.error("  ✗ TTS audio creation failed for image")
                return clip

        # Placeholder branch
        logger.warning("No media provided; using placeholder")
        placeholder = ColorClip(size=(CONTENT_WIDTH, CONTENT_HEIGHT), color=(40, 45, 55), duration=duration)
        placeholder = placeholder.set_position((CONTENT_X, CONTENT_Y))
        
        if text and text.strip():
            logger.info(f"  Placeholder segment, adding TTS for: {text[:50]}...")
            audio = _make_gtts_audio(text, language_code, duration)
            if audio:
                placeholder = placeholder.set_audio(audio)
                logger.info("  ✓ TTS audio successfully added to placeholder")
            else:
                logger.error("  ✗ TTS audio creation failed for placeholder")
        
        return placeholder
        
    except Exception as e:
        logger.error(f"process_video_segment error: {e}", exc_info=True)
        fallback = ColorClip(size=(CONTENT_WIDTH, CONTENT_HEIGHT), color=(40, 45, 55), duration=duration)
        return fallback.set_position((CONTENT_X, CONTENT_Y))

# --------------------------------------------------------------------------------------
# Core pipeline
# --------------------------------------------------------------------------------------
async def process_bulletin(bulletin_data: BulletinData) -> Dict[str, Any]:
    temp_files: List[Path] = []
    clips_to_close = []
    try:
        logger.info("\n" + "="*60)
        logger.info(f"Processing bulletin: {bulletin_data.id}")
        logger.info(f"Language: {bulletin_data.language_name} ({bulletin_data.language_code})")
        logger.info(f"Ticker: {bulletin_data.ticker[:80]}...")
        logger.info("="*60)

        # Download assets
        background_path = download_file(bulletin_data.background_url)
        if background_path:
            temp_files.append(background_path)
            logger.info("✓ Background downloaded")
        else:
            logger.warning("Background download failed; fallback color will be used")

        logo_path = download_file(bulletin_data.logo_url)
        if logo_path:
            temp_files.append(logo_path)
            logger.info("✓ Logo downloaded")
        else:
            logger.warning("Logo download failed")

        frame_path: Optional[Path] = None

        # Prep segments
        segment_rows = []
        total_duration = 0.0
        for idx, seg in enumerate(bulletin_data.content):
            logger.info(f"Segment {idx+1}: {seg.segment_type}")
            if seg.frame_url and not frame_path:
                frame_path = download_file(seg.frame_url)
                if frame_path:
                    temp_files.append(frame_path)
                    logger.info("  ✓ Frame downloaded")

            duration = seg.duration if seg.duration else 10.0
            media_path = None
            if seg.media_url:
                media_path = download_file(seg.media_url)
                if media_path:
                    temp_files.append(media_path)
                    logger.info(f"  ✓ Media downloaded: {media_path.name}")
                    if str(media_path).lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                        has_audio = check_video_has_audio(media_path)
                        logger.info(f"  Audio status: {'Has audio' if has_audio else 'No audio - will add TTS if text available'}")

            segment_rows.append({
                "segment": seg,
                "media_path": media_path,
                "duration": duration
            })
            total_duration += duration

        logger.info(f"\n📊 Total duration: {total_duration:.1f}s")
        logger.info("\n🎬 Building layers...")

        # Layer 1: Background
        background_clip = None
        if background_path and background_path.exists():
            try:
                if str(background_path).lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                    bg = VideoFileClip(str(background_path)).resize((VIDEO_WIDTH, VIDEO_HEIGHT))
                    if bg.duration < total_duration and bg.duration > 0:
                        loops = math.ceil(total_duration / bg.duration)
                        bg = concatenate_videoclips([bg] * loops, method="compose")
                    background_clip = bg.subclip(0, total_duration).without_audio()
                else:
                    img = Image.open(background_path).convert("RGB").resize((VIDEO_WIDTH, VIDEO_HEIGHT), Image.Resampling.LANCZOS)
                    background_clip = ImageClip(np.array(img), duration=total_duration)
                logger.info("  ✓ Background layer ready")
            except Exception as e:
                logger.error(f"Background error: {e}")

        if not background_clip:
            background_clip = ColorClip(size=(VIDEO_WIDTH, VIDEO_HEIGHT), color=(25, 30, 40), duration=total_duration)
            logger.info("  ✓ Fallback background ready")
        clips_to_close.append(background_clip)

        # Collect layers
        all_clips = [background_clip]
        current_time = 0.0
        valid_segments = 0

        # Layer 2: Content segments (with audio logic)
        logger.info("\n🎤 Processing segments with audio...")
        for idx, row in enumerate(segment_rows):
            dur = row["duration"]
            seg: ContentSegment = row["segment"]
            logger.info(f"  ➤ Adding segment {idx+1} at {current_time:.2f}s")
            
            # Log if TTS will be attempted
            if seg.text and seg.text.strip():
                logger.info(f"    Text for TTS: '{seg.text[:60]}...'")

            clip = process_video_segment(
                media_path=row["media_path"],
                duration=dur,
                text=seg.text,
                language_code=bulletin_data.language_code,
                loop=True
            )
            if clip:
                clip = clip.set_start(current_time)
                clips_to_close.append(clip)
                all_clips.append(clip)
                valid_segments += 1
                
                # Check if audio was added
                if clip.audio:
                    logger.info(f"    ✓ Segment {idx+1} has audio")
                else:
                    logger.info(f"    ⚠ Segment {idx+1} has no audio")
            else:
                logger.error("    ✗ Segment clip failed; using gray placeholder")
                ph = ColorClip(size=(CONTENT_WIDTH, CONTENT_HEIGHT), color=(40, 45, 55), duration=dur)
                ph = ph.set_position((CONTENT_X, CONTENT_Y)).set_start(current_time)
                clips_to_close.append(ph)
                all_clips.append(ph)

            current_time += dur

        # Layer 3: Frame
        logger.info("\n  ➤ Adding frame...")
        frame_clip = create_frame_border(frame_path, total_duration)
        clips_to_close.append(frame_clip)
        all_clips.append(frame_clip)
        logger.info("  ✓ Frame added")

        # Layer 4: Logo
        logger.info("  ➤ Adding logo...")
        logo_clip = create_logo_overlay(total_duration, logo_path)
        clips_to_close.append(logo_clip)
        all_clips.append(logo_clip)
        logger.info("  ✓ Logo added")

        # Layer 5: Ticker (opaque bar)
        logger.info("  ➤ Adding ticker...")
        ticker_clip = create_scrolling_ticker_pil(bulletin_data.ticker, total_duration)
        clips_to_close.append(ticker_clip)
        all_clips.append(ticker_clip)
        logger.info("  ✓ Ticker added")

        # Layer 6 (LAST/TOP): Top-headline overlays
        logger.info("  ➤ Adding text overlays as TOP layer...")
        current_time = 0.0
        for idx, row in enumerate(segment_rows):
            dur = row["duration"]
            seg: ContentSegment = row["segment"]
            if seg.top_headline:
                txt = create_text_overlay_pil(seg.top_headline, dur)
                if txt:
                    txt = txt.set_start(current_time)
                    clips_to_close.append(txt)
                    all_clips.append(txt)
                    logger.info(f"    ✓ Text overlay added for seg {idx+1}: '{seg.top_headline}'")
                else:
                    logger.error(f"    ✗ Text overlay failed for seg {idx+1}")
            current_time += dur

        # Composite
        logger.info("\n🎨 Compositing final video...")
        if valid_segments == 0:
            final_video = ColorClip(size=(VIDEO_WIDTH, VIDEO_HEIGHT), color=(25, 30, 40), duration=total_duration)
            clips_to_close.append(final_video)
        else:
            final_video = CompositeVideoClip(all_clips, size=(VIDEO_WIDTH, VIDEO_HEIGHT))
            
            # Check if final video has audio
            if final_video.audio:
                logger.info("  ✓ Final video has audio track")
            else:
                logger.warning("  ⚠ Final video has no audio track")

        # Thumbnail
        logger.info("\n📸 Creating thumbnail...")
        try:
            frame1 = final_video.get_frame(min(1.0, max(0.0, total_duration/2.0)))
            thumb = Image.fromarray(frame1).resize((1280, 720), Image.Resampling.LANCZOS)
        except Exception as e:
            logger.error(f"Thumbnail frame error: {e}")
            thumb = Image.new("RGB", (1280, 720), (25, 30, 40))

        thumbnail_path = THUMBNAILS_DIR / f"thumb_{bulletin_data.id}_{uuid.uuid4()}.jpg"
        thumb.save(thumbnail_path, "JPEG", quality=95)
        logger.info("  ✓ Thumbnail saved")

        # Render
        video_filename = f"bulletin_{bulletin_data.id}_{uuid.uuid4()}.mp4"
        video_path = VIDEO_BULLETIN_DIR / video_filename

        logger.info("\n💾 Rendering final video (H.264 + AAC @ CRF 20)...")
        logger.info("  Please wait, this may take a few moments...")
        
        try:
            # Ensure audio codec is set properly
            final_video.write_videofile(
                str(video_path),
                fps=24,
                codec="libx264",
                audio_codec="aac",
                audio_bitrate="128k",  # Ensure good audio quality
                preset="medium",
                ffmpeg_params=["-crf", "20", "-ar", "44100"],  # Set audio sample rate
                threads=4,
                logger=None,
                verbose=False
            )
            logger.info("  ✓ Render complete")
            
            # Verify the output has audio
            if check_video_has_audio(video_path):
                logger.info("  ✓ Output video has audio")
            else:
                logger.warning("  ⚠ Output video has no audio - check TTS generation")
                
        except Exception as e:
            logger.error(f"Render error: {e}", exc_info=True)
            raise

        # Cleanup clips
        try:
            final_video.close()
        except:
            pass
        for c in clips_to_close:
            try:
                c.close()
            except:
                pass

        # Temp + audio cleanup
        for tp in temp_files:
            _safe_unlink(tp)
        for a in AUDIO_DIR.glob("*.mp3"):
            _safe_unlink(a)

        size_mb = video_path.stat().st_size / (1024 * 1024)
        logger.info(f"\n✅ SUCCESS: {video_filename}")
        logger.info(f"📁 Size: {size_mb:.1f} MB")
        logger.info(f"⏱ Duration: {total_duration:.1f}s")
        logger.info(f"📺 Resolution: {VIDEO_WIDTH}x{VIDEO_HEIGHT}")
        logger.info("="*60 + "\n")

        return {
            "id": bulletin_data.id,
            "status": "completed",
            "video_url": f"/video-bulletin/{video_filename}",
            "thumbnail_url": f"/thumbnails/{thumbnail_path.name}",
            "video_path": str(video_path),
            "thumbnail_path": str(thumbnail_path),
            "duration": total_duration,
            "resolution": f"{VIDEO_WIDTH}x{VIDEO_HEIGHT}",
            "file_size_mb": round(size_mb, 2),
            "generated_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"\n❌ ERROR: {e}", exc_info=True)
        logger.info("="*60 + "\n")
        # Best-effort cleanup
        for c in clips_to_close:
            try: c.close()
            except: pass
        for tp in temp_files:
            _safe_unlink(tp)
        for a in AUDIO_DIR.glob("*.mp3"):
            _safe_unlink(a)
        return {
            "id": bulletin_data.id,
            "status": "failed",
            "error": str(e),
            "generated_at": datetime.now().isoformat()
        }

# --------------------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------------------
@app.post("/generate-bulletin", response_model=BulletinResponse)
async def generate_bulletin(request: BulletinRequest):
    try:
        results = []
        for b in request.data:
            out = await process_bulletin(b)
            results.append(out)
        ok = [r for r in results if r.get("status") == "completed"]
        return BulletinResponse(
            status=len(ok) > 0,
            message=f"Generated {len(ok)}/{len(request.data)} bulletins",
            data=results
        )
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/video-bulletin/{filename}")
async def get_video(filename: str):
    p = VIDEO_BULLETIN_DIR / filename
    if not p.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(path=p, media_type="video/mp4", filename=filename)

@app.get("/thumbnails/{filename}")
async def get_thumbnail(filename: str):
    p = THUMBNAILS_DIR / filename
    if not p.exists():
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    return FileResponse(path=p, media_type="image/jpeg", filename=filename)

@app.get("/")
async def root():
    return {
        "name": "News Bulletin Generator API",
        "version": "9.0.2",
        "status": "ready",
        "features": [
            "✅ FIXED: Google TTS audio now working properly",
            "✅ FIXED: Top headline position adjusted (moved up to Y=50)",
            "✅ Smart audio detection - preserves original audio when present",
            "✅ gTTS only added when no audio (with language normalization)",
            "✅ PIL-based text overlay (topmost, now properly positioned)",
            "✅ PIL-based scrolling ticker (no TextClip dependency)",
            "✅ Audio speed adjustment to fit duration",
            "✅ Frame display with proper sizing",
            "✅ Enhanced audio logging for debugging",
            "✅ Better error handling & cleanup"
        ],
        "endpoints": {
            "POST /generate-bulletin": "Generate bulletin with JSON payload",
            "GET /video-bulletin/{filename}": "Download video",
            "GET /thumbnails/{filename}": "Get thumbnail"
        },
        "json_structure": {
            "top_headline": "Text to display in upper area (Y=50)",
            "text": "Text for gTTS narration (used if video has no audio / images / placeholders)",
            "media_url": "Video or image URL",
            "frame_url": "Frame overlay image URL",
            "duration": "Segment duration in seconds"
        },
        "changes": {
            "text_position": "Headlines now at Y=50 (was 100) to fit within frame",
            "audio_handling": "Improved TTS generation with better error handling",
            "audio_quality": "Added audio bitrate and sample rate settings"
        }
    }

@app.delete("/cleanup")
async def cleanup():
    try:
        count = {"temp": 0, "old_videos": 0, "old_thumbs": 0, "audio": 0}

        for f in TEMP_DIR.glob("*"):
            try:
                f.unlink()
                count["temp"] += 1
            except:
                pass

        cutoff = datetime.now().timestamp() - 86400
        for f in VIDEO_BULLETIN_DIR.glob("*.mp4"):
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
                    count["old_videos"] += 1
            except:
                pass

        for f in THUMBNAILS_DIR.glob("*.jpg"):
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
                    count["old_thumbs"] += 1
            except:
                pass

        for f in AUDIO_DIR.glob("*.mp3"):
            try:
                f.unlink()
                count["audio"] += 1
            except:
                pass

        logger.info(f"Cleanup: {count}")
        return {"status": "success", "cleaned": count}
    except Exception as e:
        logger.error(f"Cleanup error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "="*80)
    print(" " * 20 + "NEWS BULLETIN GENERATOR v9.0.2")
    print("="*80)
    print("\n🎯 FIXES APPLIED:")
    print("  ✅ Google TTS audio generation fixed")
    print("  ✅ Top headline position adjusted to Y=50 (was 100)")
    print("  ✅ Better audio handling with proper error logging")
    print("  ✅ Audio quality settings improved")
    print("\n🎯 KEY FEATURES:")
    print("  ✅ Smart Audio Detection - Preserves original audio when present")
    print("  ✅ gTTS only for silent videos - No audio overlap")
    print("  ✅ PIL Text Overlay - Now properly positioned at top")
    print("  ✅ PIL Ticker - No ImageMagick dependency")
    print("  ✅ Audio Speed Adjustment - Fits narration to duration")
    print("\n📐 LAYOUT SPECIFICATIONS:")
    print(f"  • Output Resolution: {VIDEO_WIDTH}x{VIDEO_HEIGHT} (Full HD)")
    print(f"  • Content Area: {CONTENT_WIDTH}x{CONTENT_HEIGHT} at ({CONTENT_X}, {CONTENT_Y})")
    print(f"  • Logo Position: {LOGO_SIZE}x{LOGO_SIZE} at ({LOGO_X}, {LOGO_Y})")
    print(f"  • Text Overlay: Position ({TEXT_OVERLAY_X}, {TEXT_OVERLAY_Y})")
    print(f"  • Ticker Bar: Full width x {TICKER_HEIGHT}px at bottom")
    print("\n🔧 AUDIO HANDLING:")
    print("  • Videos WITH audio: Original audio preserved")
    print("  • Videos WITHOUT audio: gTTS narration added if text provided")
    print("  • Images & Placeholder: gTTS used if text provided")
    print("  • Smart detection prevents audio overlap")
    print("\n" + "-"*80)
    print("🚀 Starting server at http://localhost:8000")
    print("="*80 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")