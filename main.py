import os
import subprocess
import textwrap
from PIL import Image
from moviepy.editor import VideoFileClip, ImageClip, concatenate_videoclips
from mutagen.mp3 import MP3
from google.cloud import texttospeech
from pydub import AudioSegment

# Configuration
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key.json"
FONT_PATH = "assets/hindi-bold.ttf"
FRAME_IMAGE = "assets/frame.png"
BACKGROUND = "assets/bg/earth_bg.mp4"
OUTPUT_DIR = "output"

# Your content - Add more segments as needed
content = [
    {
        "media": "assets/sample_video.mp4",
        "text": "यह बड़ी खबर है जिसमें पुरातत्व विभाग की जांच चल रही है और प्रशासन ने इलाके की घेराबंदी कर दी है।"
    },
]

TICKER = "🟥 ताजा खबर: प्रशासन को चौंकाने वाली मूर्तियाँ बरामद... जांच जारी है | बड़ी खबरें सबसे पहले यहाँ! | मुख्य समाचार अभी अभी"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== Helper Functions ==========

def generate_tts(text, out_path):
    """Generate TTS audio using Google Cloud Text-to-Speech with error handling"""
    print(f"🎙️  Generating TTS for: {text[:50]}...")
    
    try:
        # Initialize client with proper error handling
        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=text)
        
        # Use high-quality Hindi voice with fallback
        voice = texttospeech.VoiceSelectionParams(
            language_code="hi-IN", 
            name="hi-IN-Wavenet-D"
        )
        
        # Configure audio output with optimal settings
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=0.85,  # Slightly slower for news clarity
            pitch=0.0,
            volume_gain_db=2.0   # Boost volume slightly
        )
        
        print("🔄 Calling Google TTS API...")
        response = client.synthesize_speech(
            input=synthesis_input, 
            voice=voice, 
            audio_config=audio_config
        )
        
        if not response.audio_content:
            raise Exception("No audio content received from Google TTS")
        
        # Save MP3 file first
        mp3_path = out_path.replace('.wav', '.mp3')
        with open(mp3_path, "wb") as out:
            out.write(response.audio_content)
        
        print(f"✅ MP3 saved: {mp3_path}")
        
        # Convert to WAV using pydub for better FFmpeg compatibility
        try:
            audio = AudioSegment.from_mp3(mp3_path)
            # Normalize audio and ensure proper format
            audio = audio.normalize()
            audio = audio.set_frame_rate(22050).set_channels(1)  # Mono, 22kHz
            audio.export(out_path, format="wav")
            print(f"✅ WAV exported: {out_path}")
        except Exception as conv_error:
            print(f"⚠️  Pydub conversion failed: {conv_error}")
            # Fallback: use FFmpeg for conversion
            subprocess.run([
                "ffmpeg", "-y", "-i", mp3_path, 
                "-acodec", "pcm_s16le", "-ar", "22050", "-ac", "1",
                out_path
            ], check=True, capture_output=True)
            print(f"✅ FFmpeg conversion successful: {out_path}")
        
        # Verify the audio file was created and has content
        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            raise Exception(f"Generated audio file is empty or doesn't exist: {out_path}")
        
        print(f"✅ TTS generation successful: {out_path}")
        return out_path
        
    except Exception as e:
        print(f"❌ TTS Generation failed: {str(e)}")
        print("🔍 Troubleshooting tips:")
        print("   1. Check if GOOGLE_APPLICATION_CREDENTIALS is set correctly")
        print("   2. Verify key.json file exists and is valid")
        print("   3. Ensure Google Cloud TTS API is enabled")
        print("   4. Check if you have sufficient API quota")
        raise e

def get_audio_duration(audio_path):
    """Get duration of audio file"""
    if audio_path.endswith('.mp3'):
        return MP3(audio_path).info.length
    else:
        audio = AudioSegment.from_wav(audio_path)
        return len(audio) / 1000.0

def prepare_media_clip(media, duration, output):
    """Prepare media for left side positioning - half screen coverage"""
    print(f"🎬 Preparing media clip: {media}")
    
    ext = os.path.splitext(media)[1].lower()
    
    # Target size for media - half screen coverage
    target_width = 960   # Half of 1920 (full left half)
    target_height = 1080  # Full height
    
    if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        # For images, create a clip with proper scaling
        clip = (ImageClip(media)
                .set_duration(duration)
                .resize((target_width, target_height))
                .set_position("center"))
                
    else:
        # For videos, resize and loop if needed
        video = VideoFileClip(media)
        
        # Calculate how many loops we need
        loops = int(duration / video.duration) + 1
        
        # Create looped video
        if loops > 1:
            looped_video = concatenate_videoclips([video] * loops)
            final_video = looped_video.subclip(0, duration)
        else:
            final_video = video.subclip(0, min(duration, video.duration))
        
        # Resize for left side with consistent dimensions
        clip = final_video.resize((target_width, target_height)).set_position("center")
    
    # Write the clip
    clip.write_videofile(output, fps=25, codec="libx264", audio=False, verbose=False, logger=None)
    clip.close()
    print(f"✅ Media clip prepared: {output}")

def create_main_layout(bg, audio, text, media_clip, out_path):
    """Create layout - media on left half, text on right half for 1920x1080"""
    print(f"🎨 Creating 1920x1080 half-screen layout...")

    # --- Layout Constants for 1920x1080 - Half Screen Layout ---
    VIDEO_WIDTH = 1920
    VIDEO_HEIGHT = 1080

    # Media positioning (left half - full coverage)
    media_width = 960
    media_height = 1080
    media_x = 0  # Start from left edge
    media_y = 0  # Start from top edge

    # Text box positioning (right half - full coverage)
    text_box_width = 960  # Right half of screen
    text_box_height = 980  # Leave space for ticker at bottom
    text_box_x = 960  # Start from middle of screen
    text_box_y = 0  # Start from top

    # Text positioning within the text box
    text_x = text_box_x + 40
    text_y = text_box_y + 100
    font_size = 64  # Much larger font

    # Headlines section
    headlines_y = text_box_y + text_box_height - 150
    headlines_height = 100

    # --- Prepare text file ---
    wrap_width = 22  # Adjusted for larger font
    wrapped = textwrap.fill(text, width=wrap_width)
    temp_text = f"{OUTPUT_DIR}/temp_text.txt"
    with open(temp_text, "w", encoding="utf-8") as f:
        f.write(wrapped)

    cmd = [
        "ffmpeg", "-y",
        "-i", bg,
        "-i", media_clip,
        "-i", audio,
        "-filter_complex",
        
        # Overlay media on left half (full coverage)
        f"[0:v][1:v]overlay={media_x}:{media_y}[media_applied];"

        # Red text background box (right half - full coverage)
        f"[media_applied]drawbox=x={text_box_x}:y={text_box_y}:w={text_box_width}:h={text_box_height}:"
        f"color=0x8B0000@1.0:t=fill[text_bg];"

        # Headlines section background
        f"[text_bg]drawbox=x={text_box_x}:y={headlines_y}:w={text_box_width}:h={headlines_height}:"
        f"color=0x660000@1.0:t=fill[headlines_bg];"

        # HEAD LINES text
        f"[headlines_bg]drawtext=fontfile='{FONT_PATH}':text='HEAD LINES':"
        f"fontcolor=white:fontsize=48:x={text_box_x + 30}:y={headlines_y + 35}:"
        f"shadowcolor=black@0.9:shadowx=3:shadowy=3[headlines_applied];"

        # Main news text with proper padding and spacing
        f"[headlines_applied]drawtext=fontfile='{FONT_PATH}':textfile='{temp_text}':"
        f"fontcolor=white:fontsize={font_size}:x={text_x}:y={text_y}:"
        f"shadowcolor=black@0.9:shadowx=3:shadowy=3:line_spacing=15[final]",

        "-map", "[final]",
        "-map", "2:a",
        "-c:v", "libx264",
        "-c:a", "aac",
        "-preset", "medium",
        "-crf", "21",
        "-shortest", out_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Layout creation error: {result.stderr}")
        raise Exception("Layout creation failed")
    
    os.remove(temp_text)
    print(f"✅ Half-screen layout generated: {out_path}")

def overlay_frame(video_with_layout, frame, out_path):
    """Overlay the frame on top of everything with proper scaling"""
    print(f"🖼️  Adding frame overlay...")
    
    cmd = [
        "ffmpeg", "-y",
        "-i", video_with_layout,
        "-i", frame,
        "-filter_complex",
        # Scale frame to match video resolution and overlay
        "[1:v]scale=1920:1080[frame_scaled];"
        "[0:v][frame_scaled]overlay=0:0[final]",
        "-map", "[final]",
        "-map", "0:a",
        "-c:v", "libx264",
        "-c:a", "aac",
        "-preset", "medium",
        "-crf", "23",
        "-shortest", out_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Frame overlay error: {result.stderr}")
        raise Exception(f"Frame overlay failed: {result.stderr}")
    
    print(f"✅ Frame overlay completed: {out_path}")

def add_ticker(input_video, text, output_video):
    """Add professional scrolling ticker at bottom with larger font"""
    print(f"📺 Adding professional ticker...")
    
    # Escape special characters for FFmpeg
    safe_text = (text.replace("\\", r"\\\\")
                    .replace(":", r"\:")
                    .replace("'", r"\'")
                    .replace("[", r"\[")
                    .replace("]", r"\]")
                    .replace("(", r"\(")
                    .replace(")", r"\)"))
    
    # Repeat text for continuous scrolling
    repeated_text = (safe_text + "     ") * 3
    
    cmd = [
        "ffmpeg", "-y",
        "-i", input_video,
        "-vf",
        # Professional ticker at bottom with larger font
        f"drawbox=y=980:color=red@1.0:width=1920:height=100:t=fill,"
        f"drawbox=y=975:color=white@0.8:width=1920:height=5:t=fill,"
        f"drawtext=fontfile='{FONT_PATH}':text='{repeated_text}':"
        f"fontsize=48:fontcolor=white:x=1920-mod(t*150\\,1920+tw):y=1020:"
        f"shadowcolor=black@0.9:shadowx=3:shadowy=3",
        "-codec:a", "copy",
        "-preset", "medium",
        output_video
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Ticker error: {result.stderr}")
        raise Exception(f"Ticker failed: {result.stderr}")
    
    print(f"✅ Professional ticker added: {output_video}")

def add_intro_and_outro(intro, main, output):
    """Add intro and outro with proper audio handling and transitions"""
    print(f"🎭 Adding intro and outro...")
    
    temp_intro = f"{OUTPUT_DIR}/intro_encoded.mp4"
    temp_outro = f"{OUTPUT_DIR}/outro_encoded.mp4"
    temp_main = f"{OUTPUT_DIR}/main_encoded.mp4"
    concat_list = f"{OUTPUT_DIR}/final_list.txt"

    # Re-encode intro with audio (keeping original audio)
    subprocess.run([
        "ffmpeg", "-y", "-i", intro, 
        "-c:v", "libx264", "-c:a", "aac", 
        "-preset", "medium", "-crf", "23",
        temp_intro
    ], check=True, capture_output=True)
    
    # Re-encode main content with audio
    subprocess.run([
        "ffmpeg", "-y", "-i", main, 
        "-c:v", "libx264", "-c:a", "aac", 
        "-preset", "medium", "-crf", "23",
        temp_main
    ], check=True, capture_output=True)
    
    # Create outro (same video as intro but muted)
    subprocess.run([
        "ffmpeg", "-y", "-i", intro, 
        "-c:v", "libx264", "-an", 
        "-preset", "medium", "-crf", "23",
        temp_outro
    ], check=True, capture_output=True)

    # Create concatenation list
    with open(concat_list, "w") as f:
        f.write(f"file '{os.path.abspath(temp_intro)}'\n")
        f.write(f"file '{os.path.abspath(temp_main)}'\n")
        f.write(f"file '{os.path.abspath(temp_outro)}'\n")

    # Concatenate all parts
    result = subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", concat_list, "-c", "copy", output
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Concatenation error: {result.stderr}")
        raise Exception(f"Concatenation failed: {result.stderr}")
    
    # Cleanup temporary files
    for temp_file in [temp_intro, temp_outro, temp_main, concat_list]:
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    print(f"✅ Intro and outro added: {output}")

# ========== Segment Generator ==========

def generate_segment(text, media_path, index):
    """Generate a single news segment with enhanced processing"""
    print(f"\n📝 Processing segment {index + 1}: {text[:30]}...")
    
    audio_path = f"{OUTPUT_DIR}/audio_{index}.wav"
    media_clip = f"{OUTPUT_DIR}/media_{index}.mp4"
    layout_video = f"{OUTPUT_DIR}/layout_{index}.mp4"
    final_segment = f"{OUTPUT_DIR}/final_{index}.mp4"

    # Generate TTS audio
    generate_tts(text, audio_path)
    duration = get_audio_duration(audio_path)
    print(f"⏱️  Segment duration: {duration:.2f} seconds")

    # Prepare media clip
    prepare_media_clip(media_path, duration, media_clip)
    
    # Create main layout with media on left and text on right
    create_main_layout(BACKGROUND, audio_path, text, media_clip, layout_video)
    
    # Overlay frame for professional look
    overlay_frame(layout_video, FRAME_IMAGE, final_segment)

    print(f"✅ Segment {index + 1} completed: {final_segment}")
    return final_segment

# ========== Main Execution Flow ==========

def main():
    """Main function to orchestrate the entire news bulletin creation"""
    print("🎬 Starting Enhanced Breaking News Bulletin Generation...")
    print(f"📊 Processing {len(content)} segments...")

    # Validate required assets
    required_assets = [FONT_PATH, FRAME_IMAGE, BACKGROUND, "assets/intro.mp4"]
    missing_assets = []
    for asset in required_assets:
        if not os.path.exists(asset):
            missing_assets.append(asset)
    
    if missing_assets:
        print(f"❌ Missing required assets: {missing_assets}")
        print("💡 Please ensure all required files are in place before running.")
        return False

    # Test Google Cloud TTS configuration
    print("🔍 Testing Google Cloud TTS configuration...")
    try:
        test_client = texttospeech.TextToSpeechClient()
        print("✅ Google Cloud TTS client initialized successfully")
    except Exception as e:
        print(f"❌ Google Cloud TTS setup error: {str(e)}")
        print("💡 Please check:")
        print("   - GOOGLE_APPLICATION_CREDENTIALS environment variable")
        print("   - key.json file exists and is valid")
        print("   - Google Cloud TTS API is enabled")
        return False

    # Generate all segments
    all_segments = []
    for idx, item in enumerate(content):
        try:
            print(f"\n{'='*60}")
            print(f"🎯 Processing Segment {idx + 1}/{len(content)}")
            print(f"📝 Text: {item['text'][:60]}...")
            print(f"🎬 Media: {item['media']}")
            print(f"{'='*60}")
            
            seg = generate_segment(item["text"], item["media"], idx)
            all_segments.append(seg)
            
            # Verify the segment was created successfully
            if not os.path.exists(seg) or os.path.getsize(seg) == 0:
                raise Exception(f"Generated segment is empty or missing: {seg}")
                
        except Exception as e:
            print(f"❌ Error processing segment {idx + 1}: {str(e)}")
            return False

    # Merge all segments
    print(f"\n🔗 Merging {len(all_segments)} segments...")
    merged_segments = f"{OUTPUT_DIR}/full_bulletin.mp4"
    concat_list = f"{OUTPUT_DIR}/concat_list.txt"
    
    with open(concat_list, "w") as f:
        for seg in all_segments:
            f.write(f"file '{os.path.abspath(seg)}'\n")

    result = subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", concat_list, "-c", "copy", merged_segments
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Merge error: {result.stderr}")
        return False

    # Add ticker
    print("📺 Adding enhanced ticker...")
    final_with_ticker = os.path.join(OUTPUT_DIR, "final_with_ticker.mp4")
    try:
        add_ticker(merged_segments, TICKER, final_with_ticker)
    except Exception as e:
        print(f"❌ Ticker addition failed: {str(e)}")
        return False

    # Add intro and outro
    print("🎭 Adding intro and outro...")
    final_output = os.path.join(OUTPUT_DIR, "final_with_intro_outro.mp4")
    try:
        add_intro_and_outro("assets/intro.mp4", final_with_ticker, final_output)
    except Exception as e:
        print(f"❌ Intro/outro addition failed: {str(e)}")
        return False

    print(f"\n🎉 SUCCESS! Final video generated: {final_output}")
    print("✅ Enhanced Breaking News Bulletin creation completed successfully!")
    
    # Display file size and duration info
    if os.path.exists(final_output):
        file_size = os.path.getsize(final_output) / (1024 * 1024)  # MB
        print(f"📁 Final file size: {file_size:.2f} MB")
        
        # Get video duration using FFmpeg
        try:
            result = subprocess.run([
                "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", final_output
            ], capture_output=True, text=True)
            if result.returncode == 0:
                duration = float(result.stdout.strip())
                print(f"⏱️  Final video duration: {duration:.1f} seconds")
        except:
            pass
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)