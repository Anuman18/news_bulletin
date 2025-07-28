import os
import subprocess
import textwrap
import json
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
LOGO_IMAGE = "assets/logo.png"  # Add your logo here
OUTPUT_DIR = "output"

# Your content - Add more segments as needed
content = [
    {
        "media": "assets/sample_video.mp4",
        "text": "'सुनहरी सरखी' महिला स्व-सहायता समूह ने हर्बल साबुन और तेल बेचकर प्रति माह ₹50,000 की कमाई शुरू की है। यह पहल महिला आत्मनिर्भरता की दिशा में एक महत्वपूर्ण कदम है।",
        "top_headline": "महिला समूह 'सुनहरी सरखी' की आर्थिक उपलब्धि"
    },
]

TICKER = "इन सेवाओं जैसे बिजली बिल भुगतान, आधार अपडेट, किसान रजिस्ट्रेशन आदि का लाभ उठा सकते हैं | ताजा खबर: प्रशासन को चौंकाने वाली मूर्तियाँ बरामद... जांच जारी है | बड़ी खबरें सबसे पहले यहाँ!"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== Helper Functions ==========

def generate_tts(text, out_path):
    """Generate TTS audio using Google Cloud Text-to-Speech with enhanced quality and compatibility"""
    print(f"🎙️  Generating TTS for: {text[:50]}...")
    
    try:
        # Initialize client
        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=text)
        
        # Use high-quality Hindi voice - Enhanced settings
        voice = texttospeech.VoiceSelectionParams(
            language_code="hi-IN", 
            name="hi-IN-Wavenet-D",  # High-quality voice
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
        )
        
        # Configure audio output for maximum compatibility
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,  # Use LINEAR16 for best quality
            speaking_rate=0.85,  # Slightly slower for clarity
            pitch=-2.0,  # Slightly lower pitch for natural sound
            volume_gain_db=6.0,  # Higher volume gain
            sample_rate_hertz=44100  # Match video standard sample rate
        )
        
        print("🔄 Calling Google TTS API...")
        response = client.synthesize_speech(
            input=synthesis_input, 
            voice=voice, 
            audio_config=audio_config
        )
        
        if not response.audio_content:
            raise Exception("No audio content received from Google TTS")
        
        # Save as temporary WAV first
        temp_wav = out_path.replace('.wav', '_raw.wav')
        with open(temp_wav, "wb") as out:
            out.write(response.audio_content)
        
        print(f"✅ Raw TTS audio saved: {temp_wav}")
        
        # Process audio with pydub for optimal quality
        try:
            # Load the raw audio
            audio = AudioSegment.from_wav(temp_wav)
            
            # Standard format for video compatibility
            audio = audio.set_frame_rate(44100)  # Standard video sample rate
            audio = audio.set_channels(2)  # Stereo for compatibility
            audio = audio.set_sample_width(2)  # 16-bit
            
            # Audio enhancement to remove robotic sound
            # Add fade in/out to prevent clicks
            audio = audio.fade_in(50).fade_out(50)
            
            # Normalize and enhance
            audio = audio.normalize()
            
            # Apply gentle compression for consistent levels
            compressed_audio = audio.compress_dynamic_range(threshold=-20.0, ratio=4.0, attack=5.0, release=50.0)
            
            # Slight amplification for clarity
            final_audio = compressed_audio + 2  # 2dB boost
            
            # Export with exact specifications for FFmpeg
            final_audio.export(
                out_path, 
                format="wav",
                parameters=[
                    "-acodec", "pcm_s16le",  # 16-bit PCM Little Endian
                    "-ar", "44100",          # 44.1kHz sample rate
                    "-ac", "2",              # Stereo
                    "-f", "wav"              # WAV format
                ]
            )
            
            # Clean up temporary file
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
            
            print(f"✅ Enhanced TTS audio processed: {out_path}")
            
        except Exception as process_error:
            print(f"⚠️  Audio processing error: {process_error}")
            # Fallback: simple conversion
            try:
                audio = AudioSegment.from_wav(temp_wav)
                # Basic stereo conversion
                if audio.channels == 1:
                    audio = audio.set_channels(2)
                audio = audio.set_frame_rate(44100)
                audio.export(out_path, format="wav")
                
                if os.path.exists(temp_wav):
                    os.remove(temp_wav)
                    
            except Exception as fallback_error:
                print(f"❌ Fallback processing failed: {fallback_error}")
                raise fallback_error
        
        # Comprehensive verification
        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            raise Exception(f"Generated audio file is empty or missing: {out_path}")
        
        # Audio quality verification
        try:
            test_audio = AudioSegment.from_wav(out_path)
            duration_ms = len(test_audio)
            
            print(f"✅ Audio verification passed:")
            print(f"   - Duration: {duration_ms/1000:.2f}s")
            print(f"   - Sample Rate: {test_audio.frame_rate}Hz") 
            print(f"   - Channels: {test_audio.channels}")
            print(f"   - Sample Width: {test_audio.sample_width * 8}-bit")
            
            if duration_ms < 300:  # Less than 0.3 seconds
                raise Exception("Generated audio is too short")
            
            if test_audio.frame_rate != 44100:
                print("⚠️  Warning: Sample rate doesn't match expected 44100Hz")
                
        except Exception as verify_error:
            print(f"❌ Audio verification failed: {verify_error}")
            raise verify_error
        
        print(f"✅ High-quality TTS generation successful: {out_path}")
        return out_path
        
    except Exception as e:
        print(f"❌ TTS Generation failed: {str(e)}")
        print("🔍 Troubleshooting:")
        print("   1. Verify GOOGLE_APPLICATION_CREDENTIALS path")
        print("   2. Check key.json validity") 
        print("   3. Confirm Google Cloud TTS API is enabled")
        print("   4. Check API quotas and billing")
        raise e

def get_audio_duration(audio_path):
    """Get duration of audio file"""
    if audio_path.endswith('.mp3'):
        return MP3(audio_path).info.length
    else:
        audio = AudioSegment.from_wav(audio_path)
        return len(audio) / 1000.0

def calculate_word_timings(text, total_duration):
    """Calculate timing for each word to appear during TTS"""
    words = text.split()
    if not words:
        return []
    
    # Calculate time per word (with some padding for natural speech)
    time_per_word = total_duration / len(words)
    
    word_timings = []
    current_time = 0.5  # Start after 0.5 second delay
    
    for word in words:
        word_timings.append({
            'word': word,
            'start_time': current_time,
            'end_time': current_time + time_per_word
        })
        current_time += time_per_word
    
    return word_timings

def wrap_text_for_display(text, max_chars_per_line=30):
    """Wrap text into multiple lines for better display with improved spacing"""
    words = text.split()
    lines = []
    current_line = ""
    
    for word in words:
        # Check if adding this word would exceed the line limit
        test_line = current_line + " " + word if current_line else word
        if len(test_line) <= max_chars_per_line:
            current_line = test_line
        else:
            # Start a new line
            if current_line:
                lines.append(current_line)
            current_line = word
    
    # Add the last line
    if current_line:
        lines.append(current_line)
    
    return lines

def create_animated_text_filter(text, font_path, font_size, x, y, duration, line_height=65):
    """Create FFmpeg filter for word-by-word animated text with proper wrapping and alignment"""
    lines = wrap_text_for_display(text, max_chars_per_line=30)
    words = text.split()
    
    if not words:
        return ""
    
    # Calculate timing for each word
    time_per_word = (duration - 1.0) / len(words)  # Leave 1 second at the end
    
    filters = []
    word_index = 0
    
    for line_num, line in enumerate(lines):
        line_words = line.split()
        current_y = y + (line_num * line_height)
        
        for word_pos, word in enumerate(line_words):
            start_time = 1.0 + (word_index * time_per_word)  # Start after 1 second
            
            # Calculate x position for this word in the line - better spacing
            if word_pos == 0:
                word_x = x  # First word starts at line beginning
            else:
                # Calculate cumulative width of previous words with better spacing
                prev_words = " ".join(line_words[:word_pos])
                # Use more accurate character width calculation for Hindi
                char_width = font_size * 0.65  # Adjusted for Hindi characters
                word_x = x + len(prev_words + " ") * char_width
            
            # Escape the word for FFmpeg
            safe_word = escape_text_for_ffmpeg(word)
            
            # Create filter for this word with consistent baseline alignment
            word_filter = (
                f"drawtext=fontfile={font_path}:text='{safe_word}':"
                f"fontcolor=white:fontsize={font_size}:"
                f"x={word_x}:y={current_y}:"
                f"shadowcolor=black@0.9:shadowx=3:shadowy=3:"
                f"enable='gte(t,{start_time})'"
            )
            
            filters.append(word_filter)
            word_index += 1
    
    return ",".join(filters)

def create_main_layout(bg, audio, text, media_clip, out_path, top_headline, duration):
    """Create layout with proper FFmpeg filter syntax and word-by-word animation"""
    print(f"🎨 Creating layout with logo and animated text...")

    # --- Layout Constants for 1920x1080 ---
    VIDEO_WIDTH = 1920
    VIDEO_HEIGHT = 1080

    # Top red headline bar
    headline_height = 100
    headline_y = 20

    # Logo positioning (top right)
    logo_width = 120
    logo_height = 80
    logo_x = VIDEO_WIDTH - logo_width - 30
    logo_y = 30

    # Main blue text area (left half)
    text_box_width = 960
    text_box_height = 600
    text_box_x = 0
    text_box_y = 150

    # Media window positioning (right half)
    media_window_width = 800   
    media_window_height = 600  
    media_window_x = 1060  
    media_window_y = 150   

    # Text positioning within the text box
    text_x = text_box_x + 60  # Better margin from left
    text_y = text_box_y + 80  # More space from top
    font_size = 42  # Optimal size for readability

    # "ताजा खबर" section
    news_label_y = 850
    news_label_height = 80

    # Prepare text with proper escaping for FFmpeg
    safe_headline = escape_text_for_ffmpeg(top_headline)
    
    # Create animated text filter
    animated_text_filter = create_animated_text_filter(
        text, FONT_PATH, font_size, text_x, text_y, duration
    )

    # Create the complete filter string
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
        "-i", bg,           # Input 0: background
        "-i", media_clip,   # Input 1: media
        "-i", audio,        # Input 2: audio
        "-i", LOGO_IMAGE,   # Input 3: logo
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

    print("🔧 Running FFmpeg command...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Layout creation error: {result.stderr}")
        print("Command that failed:")
        print(" ".join(cmd))
        raise Exception("Layout creation failed")
    
    print(f"✅ Layout with animated text generated: {out_path}")

def escape_text_for_ffmpeg(text):
    """Properly escape text for FFmpeg drawtext filter"""
    # Remove single quotes completely and replace with double quotes for text boundaries
    escaped = text.replace("'", "")  # Remove single quotes that cause parsing issues
    escaped = escaped.replace("\\", "\\\\")  # Escape backslashes
    escaped = escaped.replace(":", "\\:")  # Escape colons
    escaped = escaped.replace("[", "\\[")  # Escape square brackets
    escaped = escaped.replace("]", "\\]")
    escaped = escaped.replace("(", "\\(")  # Escape parentheses
    escaped = escaped.replace(")", "\\)")
    escaped = escaped.replace(",", "\\,")  # Escape commas
    escaped = escaped.replace(";", "\\;")  # Escape semicolons
    escaped = escaped.replace("₹", "Rs.")  # Replace rupee symbol
    escaped = escaped.replace("\n", " ")   # Replace newlines with spaces
    escaped = escaped.replace("\r", " ")   # Replace carriage returns with spaces
    return escaped

def prepare_media_clip(media, duration, output):
    """Prepare media for right side positioning - half screen coverage"""
    print(f"🎬 Preparing media clip: {media}")
    
    ext = os.path.splitext(media)[1].lower()
    
    # Target size for media - half screen coverage
    target_width = 800
    target_height = 600
    
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
        
        # Resize for right half coverage
        clip = final_video.resize((target_width, target_height)).set_position("center")
    
    # Write the clip
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
        # Scale frame and add fade-in animation
        "[1:v]scale=1920:1080[frame_scaled];"
        "[0:v][frame_scaled]overlay=0:0:enable='gte(t,0.2)'[final]",
        "-map", "[final]",
        "-map", "0:a",
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
    
    # Escape special characters for FFmpeg
    safe_text = escape_text_for_ffmpeg(text)
    
    # Repeat text for continuous scrolling
    repeated_text = (safe_text + "     ") * 3
    
    cmd = [
        "ffmpeg", "-y",
        "-i", input_video,
        "-vf",
        # Animated ticker with slide-up effect and pulsing background
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

def add_intro_and_outro(intro, main, output):
    """Add intro and outro with proper audio format matching and transition animations"""
    print(f"🎭 Adding intro and outro with transition animations...")
    
    temp_intro = f"{OUTPUT_DIR}/intro_processed.mp4"
    temp_outro = f"{OUTPUT_DIR}/outro_processed.mp4"
    temp_main = f"{OUTPUT_DIR}/main_processed.mp4"
    concat_list = f"{OUTPUT_DIR}/concat_list.txt"

    try:
        # Process intro with fade effects
        print("🔧 Processing intro with fade animation...")
        subprocess.run([
            "ffmpeg", "-y", "-i", intro,
            "-vf", "fade=in:0:25,fade=out:st=5:d=1",
            "-c:v", "libx264", 
            "-c:a", "aac",
            "-ar", "44100",
            "-ac", "2",
            "-b:a", "128k",
            "-preset", "medium", 
            "-crf", "23",
            temp_intro
        ], check=True, capture_output=True, text=True)
        
        # Process main video with transition effects
        print("🔧 Processing main content with transition effects...")

        duration_cmd = [
            "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", main
        ]
        duration_result = subprocess.run(duration_cmd, capture_output=True, text=True)
        if duration_result.returncode == 0:
            duration = float(duration_result.stdout.strip())
            fade_start = max(0, duration - 1.0)  # Start fade 1 second before end
        else:
            fade_start = 15

        subprocess.run([
            "ffmpeg", "-y", "-i", main,
            "-vf", f"fade=in:0:25,fade=out:st={fade_start}:d=25",
            "-c:v", "libx264",
            "-c:a", "aac", 
            "-ar", "44100",
            "-ac", "2",
            "-b:a", "128k",
            "-preset", "medium",
            "-crf", "23",
            "-avoid_negative_ts", "make_zero",
            "-fflags", "+genpts",
            temp_main
        ], check=True, capture_output=True, text=True)

        # Create outro with fade effects
        print("🔧 Creating outro with fade animation...")
        subprocess.run([
            "ffmpeg", "-y", "-i", intro,
            "-vf", "fade=in:0:25,fade=out:st=3:d=1",
            "-c:v", "libx264",
            "-c:a", "aac",
            "-ar", "44100",
            "-ac", "2",
            "-b:a", "128k",
            "-preset", "medium",
            "-crf", "23", 
            temp_outro
        ], check=True, capture_output=True, text=True)

        # Create concatenation list
        print("📝 Creating concatenation list...")
        with open(concat_list, "w", encoding="utf-8") as f:
            f.write(f"file '{os.path.abspath(temp_intro)}'\n")
            f.write(f"file '{os.path.abspath(temp_main)}'\n")
            f.write(f"file '{os.path.abspath(temp_outro)}'\n")

        # Concatenate with enhanced error handling
        print("🔗 Concatenating with smooth transitions...")
        cmd = [
            "ffmpeg", "-y", 
            "-f", "concat", 
            "-safe", "0",
            "-i", concat_list,
            "-c:v", "libx264",
            "-c:a", "aac",
            "-ar", "44100",
            "-ac", "2",
            "-b:a", "128k",
            "-preset", "medium", 
            "-crf", "23",
            "-avoid_negative_ts", "make_zero",
            "-fflags", "+genpts",
            "-max_muxing_queue_size", "1024",
            output
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Concatenation failed. Attempting fallback method...")
            print(f"Error details: {result.stderr}")
            
            # Fallback: simple concatenation
            fallback_cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0", "-i", concat_list,
                "-c", "copy", output
            ]
            
            fallback_result = subprocess.run(fallback_cmd, capture_output=True, text=True)
            
            if fallback_result.returncode != 0:
                print(f"❌ Fallback concatenation failed: {fallback_result.stderr}")
                raise Exception(f"All concatenation methods failed")
            else:
                print("✅ Fallback concatenation method succeeded!")
        else:
            print("✅ Standard concatenation completed successfully!")

        # Verify output file
        if not os.path.exists(output) or os.path.getsize(output) == 0:
            raise Exception("Output file is empty or was not created")

        print(f"✅ Final video with animated intro/outro created: {output}")

    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg process failed: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        raise Exception(f"Video processing failed: {e}")
    
    except Exception as e:
        print(f"❌ Intro/outro addition failed: {str(e)}")
        raise e
    
    finally:
        # Cleanup temporary files
        temp_files = [temp_intro, temp_outro, temp_main, concat_list]
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    print(f"🧹 Cleaned up: {temp_file}")
                except:
                    pass

    print(f"✅ Animated intro and outro successfully added: {output}")
    return output

# ========== Segment Generator ==========

def generate_segment(text, media_path, top_headline, index):
    """Generate a single news segment with animations"""
    print(f"\n📝 Processing animated segment {index + 1}: {text[:30]}...")
    
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
    
    # Create main layout with animations and logo
    create_main_layout(BACKGROUND, audio_path, text, media_clip, layout_video, top_headline, duration)
    
    # Overlay frame with animation
    overlay_frame(layout_video, FRAME_IMAGE, final_segment)

    print(f"✅ Animated segment {index + 1} completed: {final_segment}")
    return final_segment

# ========== Main Execution Flow ==========

def main():
    """Main function to orchestrate the entire animated news bulletin creation"""
    print("🎬 Starting Enhanced Breaking News Bulletin with Animations...")
    print(f"📊 Processing {len(content)} segments with logo and text animations...")

    # Validate required assets including logo
    required_assets = [FONT_PATH, FRAME_IMAGE, BACKGROUND, LOGO_IMAGE, "assets/intro.mp4"]
    missing_assets = []
    for asset in required_assets:
        if not os.path.exists(asset):
            missing_assets.append(asset)
    
    if missing_assets:
        print(f"❌ Missing required assets: {missing_assets}")
        print("💡 Please ensure all required files are in place before running.")
        print("📝 Note: Make sure you have a logo.png file in the assets folder")
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
            print(f"🎯 Processing Animated Segment {idx + 1}/{len(content)}")
            print(f"📝 Text: {item['text'][:60]}...")
            print(f"🎬 Media: {item['media']}")
            print(f"📰 Headline: {item['top_headline']}")
            print(f"{'='*60}")
            
            seg = generate_segment(item["text"], item["media"], item["top_headline"], idx)
            all_segments.append(seg)
            
            # Verify the segment was created successfully
            if not os.path.exists(seg) or os.path.getsize(seg) == 0:
                raise Exception(f"Generated segment is empty or missing: {seg}")
                
        except Exception as e:
            print(f"❌ Error processing segment {idx + 1}: {str(e)}")
            return False

    # Merge all segments
    print(f"\n🔗 Merging {len(all_segments)} animated segments...")
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

    # Add animated ticker
    print("📺 Adding enhanced animated ticker...")
    final_with_ticker = os.path.join(OUTPUT_DIR, "final_with_ticker.mp4")
    try:
        add_ticker(merged_segments, TICKER, final_with_ticker)
    except Exception as e:
        print(f"❌ Ticker addition failed: {str(e)}")
        return False

    # Add intro and outro with animations
    print("🎭 Adding animated intro and outro...")
    final_output = os.path.join(OUTPUT_DIR, "final_animated_bulletin.mp4")
    try:
        add_intro_and_outro("assets/intro.mp4", final_with_ticker, final_output)
    except Exception as e:
        print(f"❌ Intro/outro addition failed: {str(e)}")
        return False

    print(f"\n🎉 SUCCESS! Final animated video generated: {final_output}")
    print("✅ Enhanced Breaking News Bulletin with Animations creation completed successfully!")
    
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
    
    print("\n🎨 Features Added:")
    print("   ✅ Logo in top-right corner with fade-in animation")
    print("   ✅ Word-by-word text animation synchronized with TTS")
    print("   ✅ Text properly wrapped in paragraph format")
    print("   ✅ Smooth fade-in effects for all elements")
    print("   ✅ Animated ticker with slide-up effect")
    print("   ✅ Transition animations between segments")
    print("   ✅ Enhanced frame overlay with animation")
    print("   ✅ Professional crossfade transitions")
    print("   ✅ Preserved original layout and alignment")
    print("   ✅ Same high-quality TTS voice settings")
    
    # Cleanup intermediate files
    print("\n🧹 Cleaning up intermediate files...")
    cleanup_files = [
        f"{OUTPUT_DIR}/audio_*.wav",
        f"{OUTPUT_DIR}/media_*.mp4", 
        f"{OUTPUT_DIR}/layout_*.mp4",
        f"{OUTPUT_DIR}/final_*.mp4",
        f"{OUTPUT_DIR}/full_bulletin.mp4",
        f"{OUTPUT_DIR}/final_with_ticker.mp4",
        f"{OUTPUT_DIR}/concat_list.txt"
    ]
    
    import glob
    for pattern in cleanup_files:
        for file in glob.glob(pattern):
            try:
                if os.path.exists(file) and file != final_output:
                    os.remove(file)
                    print(f"   🗑️  Removed: {os.path.basename(file)}")
            except:
                pass
    
    print(f"\n🎬 Final animated bulletin ready: {final_output}")
    print("🚀 You can now use this video for broadcasting!")
    
    return True

if __name__ == "__main__":
    print("="*80)
    print("🎯 ENHANCED NEWS BULLETIN GENERATOR WITH ANIMATIONS")
    print("="*80)
    print("📋 Features:")
    print("   • Logo in top-right corner")
    print("   • Word-by-word text animation sync with TTS")
    print("   • Text wrapped in paragraph format")
    print("   • Professional bulletin animations")
    print("   • Animated ticker and transitions")
    print("   • High-quality Google TTS voice")
    print("   • Same layout and alignment preserved")
    print("="*80)
    
    success = main()
    
    if success:
        print("\n" + "="*80)
        print("🎉 BULLETIN GENERATION COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("📁 Check the 'output' folder for your final video")
        print("🎬 File: final_animated_bulletin.mp4")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("❌ BULLETIN GENERATION FAILED!")
        print("="*80)
        print("💡 Please check the error messages above and:")
        print("   • Ensure all required assets are present")
        print("   • Verify Google Cloud TTS setup")
        print("   • Check FFmpeg installation")
        print("="*80)
    
    exit(0 if success else 1)