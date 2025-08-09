import os
import json
import shutil
import asyncio
from typing import List, Dict, Optional, Any
from datetime import datetime
from pathlib import Path
import uuid
import base64
from io import BytesIO
import math

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

try:
    from gtts import gTTS
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    import requests
    from moviepy.editor import *
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install: pip install fastapi uvicorn gtts opencv-python-headless pillow numpy moviepy requests")
    exit(1)

app = FastAPI(title="News Bulletin Generator API", version="8.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent
THUMBNAILS_DIR = BASE_DIR / "thumbnails"
VIDEO_BULLETIN_DIR = BASE_DIR / "video-bulletin"
TEMP_DIR = BASE_DIR / "temp"
FONTS_DIR = BASE_DIR / "fonts"

for dir_path in [THUMBNAILS_DIR, VIDEO_BULLETIN_DIR, TEMP_DIR, FONTS_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# Video dimensions
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080

# Frame dimensions - properly aligned content area
CONTENT_X = 110  # Left margin for frame
CONTENT_Y = 180  # Top margin (below headline area)
CONTENT_WIDTH = 1700  # Width of content area
CONTENT_HEIGHT = 700  # Height of content area (leaving space for ticker)

# Ticker dimensions - fixed positioning
TICKER_HEIGHT = 120  # Increased height for better visibility
TICKER_Y = VIDEO_HEIGHT - TICKER_HEIGHT

# Headline area - properly aligned
HEADLINE_X = 110
HEADLINE_Y = 50
HEADLINE_WIDTH = 500  # Increased width for better text fit
HEADLINE_HEIGHT = 100

# Logo area - properly aligned
LOGO_SIZE = 150
LOGO_X = VIDEO_WIDTH - LOGO_SIZE - 80  # Better positioning
LOGO_Y = 30  # Slightly raised

# Hindi font URLs
HINDI_FONT_URL = "https://github.com/google/fonts/raw/main/ofl/notosansdevanagari/NotoSansDevanagari%5Bwdth%2Cwght%5D.ttf"
HINDI_FONT_PATH = FONTS_DIR / "hindi-bold.ttf"
BOLD_FONT_URL = "https://github.com/google/fonts/raw/main/apache/robotocondensed/RobotoCondensed-Bold.ttf"
BOLD_FONT_PATH = FONTS_DIR / "roboto-bold.ttf"

def setup_fonts():
    """Download fonts if not present"""
    try:
        # Download Hindi font
        if not HINDI_FONT_PATH.exists():
            print("Downloading Hindi font...")
            response = requests.get(HINDI_FONT_URL, timeout=30)
            response.raise_for_status()
            with open(HINDI_FONT_PATH, 'wb') as f:
                f.write(response.content)
            print("Hindi font downloaded")
        
        # Download bold font for ticker
        if not BOLD_FONT_PATH.exists():
            print("Downloading Roboto bold font...")
            response = requests.get(BOLD_FONT_URL, timeout=30)
            response.raise_for_status()
            with open(BOLD_FONT_PATH, 'wb') as f:
                f.write(response.content)
            print("Roboto bold font downloaded")
    except Exception as e:
        print(f"Font download warning: {e}")

setup_fonts()

class ContentSegment(BaseModel):
    segment_type: str
    media_url: Optional[str] = None
    text: Optional[str] = None  # For voice only, not displayed
    top_headline: Optional[str] = None
    frame_url: Optional[str] = None

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

def download_file(url: str, destination: Path = TEMP_DIR) -> Optional[Path]:
    """Download file from URL"""
    try:
        if not url:
            return None
            
        if url.startswith('data:'):
            header, data = url.split(',', 1)
            file_data = base64.b64decode(data)
            ext = '.png' if 'png' in header else '.jpg'
            file_path = destination / f"{uuid.uuid4()}{ext}"
            with open(file_path, 'wb') as f:
                f.write(file_data)
            return file_path
        
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        content_type = response.headers.get('content-type', '')
        if 'video' in content_type:
            ext = '.mp4'
        elif 'png' in content_type:
            ext = '.png'
        elif 'jpeg' in content_type or 'jpg' in content_type:
            ext = '.jpg'
        else:
            ext = Path(url.split('?')[0]).suffix or '.mp4'
        
        file_path = destination / f"{uuid.uuid4()}{ext}"
        
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        return file_path
    except Exception as e:
        print(f"Download failed for {url}: {e}")
        return None

def generate_tts_hindi(text: str) -> Optional[Path]:
    """Generate Hindi TTS audio (voice only, not displayed)"""
    try:
        if not text or not text.strip():
            return None
            
        tts = gTTS(text=text.strip(), lang='hi', slow=False)
        audio_path = TEMP_DIR / f"tts_{uuid.uuid4()}.mp3"
        tts.save(str(audio_path))
        return audio_path
    except Exception as e:
        print(f"TTS failed: {e}")
        return None

def create_headline_and_logo_overlay(duration: float, headline_text: Optional[str], 
                                     logo_path: Optional[Path]) -> VideoFileClip:
    """Create properly aligned headline and logo overlay"""
    try:
        # Create transparent overlay
        overlay_img = Image.new('RGBA', (VIDEO_WIDTH, VIDEO_HEIGHT), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay_img)
        
        # Add headline block with better styling
        if headline_text and headline_text.strip():
            try:
                # Draw gradient background for headline
                for i in range(HEADLINE_HEIGHT):
                    alpha = int(240 - i * 0.5)  # Gradient effect
                    draw.rectangle([HEADLINE_X, HEADLINE_Y + i, 
                                  HEADLINE_X + HEADLINE_WIDTH, HEADLINE_Y + i + 1],
                                  fill=(255, 255, 255, alpha))
                
                # Draw border for headline
                draw.rectangle([HEADLINE_X, HEADLINE_Y, 
                              HEADLINE_X + HEADLINE_WIDTH, HEADLINE_Y + HEADLINE_HEIGHT],
                              outline=(220, 20, 60, 255),  # Red border
                              width=3)
                
                # Load font for Hindi text
                try:
                    if HINDI_FONT_PATH.exists():
                        font = ImageFont.truetype(str(HINDI_FONT_PATH), 42)
                    else:
                        font = ImageFont.truetype("Arial-Bold", 42)
                except:
                    font = ImageFont.load_default()
                
                # Calculate text position for better centering
                text = headline_text.strip()
                try:
                    bbox = draw.textbbox((0, 0), text, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                except:
                    text_width, text_height = draw.textsize(text, font=font)
                
                text_x = HEADLINE_X + (HEADLINE_WIDTH - text_width) // 2
                text_y = HEADLINE_Y + (HEADLINE_HEIGHT - text_height) // 2
                
                # Draw shadow for better readability
                draw.text((text_x + 2, text_y + 2), text, fill=(100, 100, 100, 180), font=font)
                # Draw main text
                draw.text((text_x, text_y), text, fill=(220, 20, 60), font=font)  # Red text
                
                print(f"  Headline added: {headline_text}")
            except Exception as e:
                print(f"  Headline error: {e}")
        
        # Add logo with proper alignment and background
        if logo_path and logo_path.exists():
            try:
                # Create white background circle for logo
                logo_bg_size = LOGO_SIZE + 20
                draw.ellipse([LOGO_X - 10, LOGO_Y - 10, 
                            LOGO_X + logo_bg_size - 10, LOGO_Y + logo_bg_size - 10],
                            fill=(255, 255, 255, 230),
                            outline=(220, 20, 60, 255),
                            width=3)
                
                # Load and resize logo
                logo = Image.open(logo_path).convert('RGBA')
                logo_ratio = min(LOGO_SIZE / logo.width, LOGO_SIZE / logo.height)
                new_width = int(logo.width * logo_ratio * 0.9)  # Slightly smaller for padding
                new_height = int(logo.height * logo_ratio * 0.9)
                
                logo = logo.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Center logo in the circle
                logo_x = LOGO_X + (LOGO_SIZE - new_width) // 2
                logo_y = LOGO_Y + (LOGO_SIZE - new_height) // 2
                overlay_img.paste(logo, (logo_x, logo_y), logo)
                
                print(f"  Logo added: {new_width}x{new_height} at ({logo_x}, {logo_y})")
            except Exception as e:
                print(f"  Logo error: {e}")
        
        # Convert to video clip
        overlay_array = np.array(overlay_img)
        return ImageClip(overlay_array, transparent=True, duration=duration)
        
    except Exception as e:
        print(f"Overlay creation error: {e}")
        transparent = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 4), dtype=np.uint8)
        return ImageClip(transparent, transparent=True, duration=duration)

def create_frame_border(frame_path: Optional[Path], duration: float) -> VideoFileClip:
    """Create properly aligned frame border"""
    try:
        if frame_path and frame_path.exists():
            # Load and use the provided frame
            frame_img = Image.open(frame_path).convert('RGBA')
            frame_img = frame_img.resize((VIDEO_WIDTH, VIDEO_HEIGHT), Image.Resampling.LANCZOS)
        else:
            # Create default frame with professional borders
            frame_img = Image.new('RGBA', (VIDEO_WIDTH, VIDEO_HEIGHT), (0, 0, 0, 0))
            draw = ImageDraw.Draw(frame_img)
            
            # Outer golden frame
            draw.rectangle([CONTENT_X - 10, CONTENT_Y - 10, 
                           CONTENT_X + CONTENT_WIDTH + 10, CONTENT_Y + CONTENT_HEIGHT + 10],
                          outline=(255, 215, 0, 255), width=10)
            
            # Inner white accent
            draw.rectangle([CONTENT_X - 5, CONTENT_Y - 5, 
                           CONTENT_X + CONTENT_WIDTH + 5, CONTENT_Y + CONTENT_HEIGHT + 5],
                          outline=(255, 255, 255, 200), width=3)
            
            # Corner accents
            corner_size = 50
            corner_width = 5
            # Top-left
            draw.line([(CONTENT_X - 10, CONTENT_Y - 10), 
                      (CONTENT_X - 10 + corner_size, CONTENT_Y - 10)], 
                     fill=(220, 20, 60), width=corner_width)
            draw.line([(CONTENT_X - 10, CONTENT_Y - 10), 
                      (CONTENT_X - 10, CONTENT_Y - 10 + corner_size)], 
                     fill=(220, 20, 60), width=corner_width)
            # Top-right
            draw.line([(CONTENT_X + CONTENT_WIDTH + 10 - corner_size, CONTENT_Y - 10), 
                      (CONTENT_X + CONTENT_WIDTH + 10, CONTENT_Y - 10)], 
                     fill=(220, 20, 60), width=corner_width)
            draw.line([(CONTENT_X + CONTENT_WIDTH + 10, CONTENT_Y - 10), 
                      (CONTENT_X + CONTENT_WIDTH + 10, CONTENT_Y - 10 + corner_size)], 
                     fill=(220, 20, 60), width=corner_width)
        
        frame_array = np.array(frame_img)
        return ImageClip(frame_array, transparent=True, duration=duration)
        
    except Exception as e:
        print(f"Frame border error: {e}")
        transparent = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 4), dtype=np.uint8)
        return ImageClip(transparent, transparent=True, duration=duration)

def create_scrolling_ticker(ticker_text: str, duration: float) -> VideoFileClip:
    """Create high-visibility scrolling ticker with better font"""
    try:
        # Create ticker background with gradient
        ticker_bg = ColorClip(size=(VIDEO_WIDTH, TICKER_HEIGHT), 
                             color=(180, 20, 30), duration=duration)
        
        # Add gradient overlay for depth
        gradient_img = Image.new('RGBA', (VIDEO_WIDTH, TICKER_HEIGHT), (0, 0, 0, 0))
        draw = ImageDraw.Draw(gradient_img)
        for i in range(TICKER_HEIGHT // 2):
            alpha = int(100 - i * 2)
            draw.rectangle([0, i, VIDEO_WIDTH, i + 1],
                          fill=(255, 255, 255, alpha))
        gradient_overlay = ImageClip(np.array(gradient_img), transparent=True, duration=duration)
        
        # Create scrolling text with better visibility
        scroll_text = f"  ●  {ticker_text}  ●  {ticker_text}  ●  {ticker_text}  ●  "
        
        # Create text clip with better font
        try:
            # Try to use downloaded bold font first
            if BOLD_FONT_PATH.exists():
                font_path = str(BOLD_FONT_PATH)
            elif HINDI_FONT_PATH.exists():
                font_path = str(HINDI_FONT_PATH)
            else:
                font_path = 'Arial-Bold'
            
            text_clip = TextClip(
                scroll_text,
                fontsize=55,  # Larger font size
                color='white',
                font=font_path,
                stroke_color='black',  # Add black stroke for better visibility
                stroke_width=2,
                method='label'
            ).set_duration(duration)
        except:
            # Fallback with basic settings
            text_clip = TextClip(
                scroll_text,
                fontsize=55,
                color='white',
                font='Arial',
                method='label'
            ).set_duration(duration)
        
        # Calculate scrolling animation
        text_width = text_clip.w
        scroll_speed = 150  # Increased speed for smoother scroll
        
        # Lambda function for continuous scrolling
        def scroll_position(t):
            x = VIDEO_WIDTH - int((t * scroll_speed) % (text_width + VIDEO_WIDTH))
            y = (TICKER_HEIGHT - 55) // 2
            return (x, y)
        
        # Apply scrolling animation
        scrolling_text = text_clip.set_position(scroll_position)
        
        # Create BREAKING NEWS label with better styling
        label_width = 200
        label_height = 80
        
        # Create label background with gradient
        label_img = Image.new('RGBA', (label_width, label_height), (255, 255, 255, 255))
        label_draw = ImageDraw.Draw(label_img)
        
        # Add red accent
        label_draw.rectangle([0, 0, label_width, 10], fill=(220, 20, 60))
        label_draw.rectangle([0, label_height - 10, label_width, label_height], fill=(220, 20, 60))
        
        label_bg = ImageClip(np.array(label_img), duration=duration)
        
        # Create BREAKING text
        breaking_text = TextClip("BREAKING", fontsize=32, color='red', 
                               font='Arial-Bold', method='label')
        breaking_text = breaking_text.set_duration(duration).set_position(('center', 'center'))
        
        news_text = TextClip("NEWS", fontsize=24, color='black', 
                           font='Arial-Bold', method='label')
        news_text = news_text.set_duration(duration).set_position(('center', 45))
        
        # Composite label
        label = CompositeVideoClip([label_bg, breaking_text, news_text], 
                                  size=(label_width, label_height))
        label = label.set_position((20, (TICKER_HEIGHT - label_height) // 2))
        
        # Add flashing effect to label
        label = label.crossfadein(0.5).crossfadeout(0.5)
        
        # Composite all ticker elements
        ticker = CompositeVideoClip([ticker_bg, gradient_overlay, scrolling_text, label],
                                   size=(VIDEO_WIDTH, TICKER_HEIGHT))
        
        return ticker.set_position((0, TICKER_Y))
        
    except Exception as e:
        print(f"Ticker error: {e}")
        # Return basic ticker as fallback
        return ColorClip(size=(VIDEO_WIDTH, TICKER_HEIGHT), 
                        color=(180, 20, 30), duration=duration).set_position((0, TICKER_Y))

def process_video_segment(media_path: Optional[Path], duration: float, loop: bool = True) -> VideoFileClip:
    """Process video segment with proper looping"""
    try:
        if media_path and media_path.exists():
            if str(media_path).lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                # Load video
                video = VideoFileClip(str(media_path))
                
                # Loop video to match audio duration
                if loop and video.duration < duration:
                    loops_needed = math.ceil(duration / video.duration)
                    video_list = [video] * loops_needed
                    video = concatenate_videoclips(video_list)
                    print(f"    Video looped {loops_needed} times to match {duration:.1f}s duration")
                
                # Trim to exact duration
                video = video.subclip(0, duration)
                
                # Resize to fit exactly within content area
                video = video.resize((CONTENT_WIDTH, CONTENT_HEIGHT))
                
                # Position at exact content coordinates
                video = video.set_position((CONTENT_X, CONTENT_Y))
                
                print(f"    Video positioned at ({CONTENT_X}, {CONTENT_Y}), size: {CONTENT_WIDTH}x{CONTENT_HEIGHT}")
                
                return video
            else:
                # Handle image - create video from static image
                img = Image.open(media_path)
                img = img.resize((CONTENT_WIDTH, CONTENT_HEIGHT), Image.Resampling.LANCZOS)
                img_array = np.array(img)
                
                if len(img_array.shape) == 2:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
                elif img_array.shape[2] == 4:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
                
                # Add subtle zoom effect for images
                clip = ImageClip(img_array, duration=duration)
                clip = clip.set_position((CONTENT_X, CONTENT_Y))
                
                # Optional: add zoom effect
                clip = clip.resize(lambda t: 1 + 0.02 * t)  # Slow zoom
                clip = clip.set_position((CONTENT_X, CONTENT_Y))
                
                return clip
        else:
            # Default placeholder with animation
            clip = ColorClip(size=(CONTENT_WIDTH, CONTENT_HEIGHT), 
                           color=(40, 45, 55), duration=duration)
            clip = clip.set_position((CONTENT_X, CONTENT_Y))
            return clip
            
    except Exception as e:
        print(f"Video processing error: {e}")
        clip = ColorClip(size=(CONTENT_WIDTH, CONTENT_HEIGHT), 
                       color=(40, 45, 55), duration=duration)
        clip = clip.set_position((CONTENT_X, CONTENT_Y))
        return clip

async def process_bulletin(bulletin_data: BulletinData) -> Dict[str, Any]:
    """Process a single bulletin with all fixes"""
    temp_files = []
    clips_to_close = []
    
    try:
        print(f"\n{'='*60}")
        print(f"Processing bulletin: {bulletin_data.id}")
        print(f"Language: {bulletin_data.language_name}")
        print(f"Ticker: {bulletin_data.ticker[:80]}...")
        print(f"{'='*60}")
        
        # Download resources
        background_path = download_file(bulletin_data.background_url)
        if background_path:
            temp_files.append(background_path)
            print("✓ Background downloaded")
        
        logo_path = download_file(bulletin_data.logo_url)
        if logo_path:
            temp_files.append(logo_path)
            print("✓ Logo downloaded")
        
        # Process segments
        segment_data = []
        total_duration = 0
        frame_path = None
        headline_text = None
        
        for i, segment in enumerate(bulletin_data.content):
            print(f"\nSegment {i+1}: {segment.segment_type}")
            
            # Get frame from first segment that has it
            if segment.frame_url and not frame_path:
                frame_path = download_file(segment.frame_url)
                if frame_path:
                    temp_files.append(frame_path)
                    print("  ✓ Frame downloaded")
            
            # Get headline from first segment that has it
            if segment.top_headline and not headline_text:
                headline_text = segment.top_headline
                print(f"  ✓ Headline: {headline_text}")
            
            # Generate TTS for voiceover (text is for voice only, not displayed)
            audio_path = None
            audio_duration = 10.0  # Default duration if no text
            
            if segment.text and segment.text.strip():
                print(f"  ⚡ Generating voiceover (text won't be displayed)...")
                audio_path = generate_tts_hindi(segment.text)
                if audio_path:
                    temp_files.append(audio_path)
                    try:
                        audio_clip = AudioFileClip(str(audio_path))
                        audio_duration = audio_clip.duration
                        audio_clip.close()
                        print(f"  ✓ Voiceover generated: {audio_duration:.1f}s")
                    except:
                        pass
            
            # Download media
            media_path = None
            if segment.media_url:
                media_path = download_file(segment.media_url)
                if media_path:
                    temp_files.append(media_path)
                    print(f"  ✓ Media downloaded")
            
            segment_data.append({
                'segment': segment,
                'audio_path': audio_path,
                'media_path': media_path,
                'duration': audio_duration
            })
            
            total_duration += audio_duration
        
        print(f"\n📊 Total duration: {total_duration:.1f}s")
        
        # Create base layers
        print("\n🎬 Creating video layers...")
        
        # 1. Background layer
        if background_path and background_path.exists():
            if str(background_path).lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                bg = VideoFileClip(str(background_path))
                bg = bg.resize((VIDEO_WIDTH, VIDEO_HEIGHT))
                if bg.duration < total_duration:
                    loops = math.ceil(total_duration / bg.duration)
                    bg = concatenate_videoclips([bg] * loops)
                bg = bg.subclip(0, total_duration)
            else:
                img = Image.open(background_path)
                img = img.resize((VIDEO_WIDTH, VIDEO_HEIGHT), Image.Resampling.LANCZOS)
                bg = ImageClip(np.array(img), duration=total_duration)
        else:
            bg = ColorClip(size=(VIDEO_WIDTH, VIDEO_HEIGHT), 
                         color=(25, 30, 40), duration=total_duration)
        clips_to_close.append(bg)
        print("  ✓ Background layer created")
        
        # Start building composite
        all_clips = [bg]
        
        # 2. Process video segments with looping
        current_time = 0
        all_audio = []
        
        for i, seg_data in enumerate(segment_data):
            duration = seg_data['duration']
            
            print(f"  ➤ Processing segment {i+1}...")
            
            # Add video with looping to match audio duration
            video_clip = process_video_segment(seg_data['media_path'], duration, loop=True)
            video_clip = video_clip.set_start(current_time)
            clips_to_close.append(video_clip)
            all_clips.append(video_clip)
            print(f"    ✓ Video added (looped to {duration:.1f}s)")
            
            # Add audio (voice only, no text on screen)
            if seg_data['audio_path']:
                try:
                    audio = AudioFileClip(str(seg_data['audio_path']))
                    audio = audio.set_start(current_time)
                    all_audio.append(audio)
                    clips_to_close.append(audio)
                    print(f"    ✓ Voiceover added (text not displayed)")
                except Exception as e:
                    print(f"    ⚠ Audio error: {e}")
            
            current_time += duration
        
        # 3. Add frame border
        print("  ➤ Adding frame border...")
        frame_border = create_frame_border(frame_path, total_duration)
        clips_to_close.append(frame_border)
        all_clips.append(frame_border)
        print("  ✓ Frame border added")
        
        # 4. Add headline and logo overlay
        print("  ➤ Adding headline and logo...")
        headline_logo_overlay = create_headline_and_logo_overlay(total_duration, headline_text, logo_path)
        clips_to_close.append(headline_logo_overlay)
        all_clips.append(headline_logo_overlay)
        print("  ✓ Headline and logo added")
        
        # 5. Add ticker with better visibility
        print("  ➤ Adding ticker...")
        ticker = create_scrolling_ticker(bulletin_data.ticker, total_duration)
        clips_to_close.append(ticker)
        all_clips.append(ticker)
        print("  ✓ Ticker added with improved visibility")
        
        # Composite all layers
        print("\n🎨 Compositing final video...")
        final_video = CompositeVideoClip(all_clips, size=(VIDEO_WIDTH, VIDEO_HEIGHT))
        
        # Add audio track
        if all_audio:
            final_audio = CompositeAudioClip(all_audio)
            final_video = final_video.set_audio(final_audio)
            print("  ✓ Audio track added")
        
        # Generate thumbnail
        print("📸 Creating thumbnail...")
        try:
            first_frame = final_video.get_frame(1)
            thumbnail = Image.fromarray(first_frame)
            thumbnail = thumbnail.resize((1280, 720), Image.Resampling.LANCZOS)
        except:
            thumbnail = Image.new('RGB', (1280, 720), (25, 30, 40))
        
        thumbnail_path = THUMBNAILS_DIR / f"thumb_{bulletin_data.id}_{uuid.uuid4()}.jpg"
        thumbnail.save(thumbnail_path, 'JPEG', quality=95)
        print("  ✓ Thumbnail created")
        
        # Save video
        video_filename = f"bulletin_{bulletin_data.id}_{uuid.uuid4()}.mp4"
        video_path = VIDEO_BULLETIN_DIR / video_filename
        
        print("\n💾 Rendering final video...")
        print("  This may take a few minutes...")
        final_video.write_videofile(
            str(video_path),
            fps=24,
            codec='libx264',
            audio_codec='aac',
            preset='medium',
            ffmpeg_params=['-crf', '20'],  # Better quality
            threads=4,
            logger=None,
            verbose=False
        )
        
        # Cleanup
        final_video.close()
        for clip in clips_to_close:
            try:
                clip.close()
            except:
                pass
        
        for temp_file in temp_files:
            try:
                if temp_file and temp_file.exists():
                    temp_file.unlink()
            except:
                pass
        
        file_size = video_path.stat().st_size / (1024 * 1024)
        print(f"\n✅ SUCCESS: {video_filename}")
        print(f"📁 Size: {file_size:.1f} MB")
        print(f"⏱ Duration: {total_duration:.1f}s")
        print(f"📺 Resolution: {VIDEO_WIDTH}x{VIDEO_HEIGHT}")
        print(f"{'='*60}\n")
        
        return {
            "id": bulletin_data.id,
            "status": "completed",
            "video_url": f"/video-bulletin/{video_filename}",
            "thumbnail_url": f"/thumbnails/{thumbnail_path.name}",
            "video_path": str(video_path),
            "thumbnail_path": str(thumbnail_path),
            "duration": total_duration,
            "resolution": f"{VIDEO_WIDTH}x{VIDEO_HEIGHT}",
            "file_size_mb": round(file_size, 2),
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print(f"{'='*60}\n")
        
        # Cleanup on error
        for clip in clips_to_close:
            try:
                clip.close()
            except:
                pass
        
        for temp_file in temp_files:
            try:
                if temp_file and temp_file.exists():
                    temp_file.unlink()
            except:
                pass
        
        return {
            "id": bulletin_data.id,
            "status": "failed",
            "error": str(e),
            "generated_at": datetime.now().isoformat()
        }

@app.post("/generate-bulletin")
async def generate_bulletin(request: BulletinRequest):
    """Generate news bulletin from JSON data"""
    try:
        results = []
        
        for bulletin_data in request.data:
            result = await process_bulletin(bulletin_data)
            results.append(result)
        
        successful = [r for r in results if r.get('status') == 'completed']
        
        return BulletinResponse(
            status=len(successful) > 0,
            message=f"Generated {len(successful)}/{len(request.data)} bulletins",
            data=results
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/video-bulletin/{filename}")
async def get_video(filename: str):
    """Download generated video"""
    file_path = VIDEO_BULLETIN_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(path=file_path, media_type="video/mp4", filename=filename)

@app.get("/thumbnails/{filename}")
async def get_thumbnail(filename: str):
    """Get video thumbnail"""
    file_path = THUMBNAILS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    return FileResponse(path=file_path, media_type="image/jpeg", filename=filename)

@app.get("/")
async def root():
    """API information endpoint"""
    return {
        "name": "News Bulletin Generator API",
        "version": "8.0.0",
        "status": "ready",
        "improvements": [
            "✅ Ticker text visibility fixed with better font and contrast",
            "✅ Frame properly aligned with content area",
            "✅ Logo properly aligned with background circle",
            "✅ Text used ONLY for voice (TTS) - not displayed on screen",
            "✅ Video loops automatically to match audio duration",
            "✅ Headline text properly styled and positioned",
            "✅ Professional news channel appearance",
            "✅ Better visual hierarchy and readability"
        ],
        "endpoints": {
            "POST /generate-bulletin": "Generate bulletin with JSON payload",
            "GET /video-bulletin/{filename}": "Download video",
            "GET /thumbnails/{filename}": "Get thumbnail"
        },
        "test_with_postman": {
            "url": "http://localhost:8000/generate-bulletin",
            "method": "POST",
            "headers": {
                "Content-Type": "application/json"
            },
            "body": "Use the JSON below in raw body"
        }
    }

@app.delete("/cleanup")
async def cleanup():
    """Clean up temporary files"""
    try:
        count = {"temp": 0, "old_videos": 0, "old_thumbs": 0}
        
        # Clean temp directory
        for f in TEMP_DIR.glob("*"):
            try:
                f.unlink()
                count["temp"] += 1
            except:
                pass
        
        # Clean old files (older than 24 hours)
        cutoff = datetime.now().timestamp() - 86400
        
        for f in VIDEO_BULLETIN_DIR.glob("*.mp4"):
            if f.stat().st_mtime < cutoff:
                try:
                    f.unlink()
                    count["old_videos"] += 1
                except:
                    pass
        
        for f in THUMBNAILS_DIR.glob("*.jpg"):
            if f.stat().st_mtime < cutoff:
                try:
                    f.unlink()
                    count["old_thumbs"] += 1
                except:
                    pass
        
        return {"status": "success", "cleaned": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("\n" + "="*80)
    print(" " * 20 + "NEWS BULLETIN GENERATOR v8.0.0")
    print("="*80)
    print("\n🎯 ALL ISSUES FIXED:")
    print("  ✅ Ticker text now visible with high contrast")
    print("  ✅ Frame properly aligned around content")
    print("  ✅ Logo properly positioned with background")
    print("  ✅ Text used ONLY for voice - not displayed")
    print("  ✅ Video loops to match audio duration")
    print("  ✅ Professional news channel appearance")
    print("\n📐 LAYOUT SPECIFICATIONS:")
    print(f"  • Output Resolution: {VIDEO_WIDTH}x{VIDEO_HEIGHT} (Full HD)")
    print(f"  • Content Area: {CONTENT_WIDTH}x{CONTENT_HEIGHT} at ({CONTENT_X}, {CONTENT_Y})")
    print(f"  • Headline Box: {HEADLINE_WIDTH}x{HEADLINE_HEIGHT} at ({HEADLINE_X}, {HEADLINE_Y})")
    print(f"  • Logo Position: {LOGO_SIZE}x{LOGO_SIZE} at ({LOGO_X}, {LOGO_Y})")
    print(f"  • Ticker Bar: Full width x {TICKER_HEIGHT}px at bottom")
    print("\n🔧 KEY FEATURES:")
    print("  • Hindi TTS voiceover generation")
    print("  • Automatic video looping")
    print("  • Scrolling ticker with high visibility")
    print("  • Professional frame borders")
    print("  • Clean, broadcast-quality output")
    print("\n" + "-"*80)
    print("🚀 Starting server at http://localhost:8000")
    print("📝 Use Postman to test with the JSON payload below")
    print("="*80 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")