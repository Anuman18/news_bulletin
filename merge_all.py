import subprocess
import os
from scrolling_ticker import add_scrolling_ticker
from eft_text_tts import generate_google_tts, add_left_text_with_box

# Paths
intro_path = "output/intro.mp4"
intro_audio_path = "output/n_intro_with_audio.mp4"
left_path = "output/left_text_with_tts.mp4"
left_fixed_path = "output/left_text_with_tts_1080p.mp4"
merged_output = "output/full_bulletin.mp4"
final_with_ticker = "output/final_with_ticker_1.mp4"
font_path = "assets/hindi-bold.ttf"
ticker_text = "🔴 ताजा खबर: प्रशासन को चौंकाने वाली मूर्तियाँ बरामद... जांच जारी है | बड़ी खबरें सबसे पहले यहाँ!"

# Text for left-side box
input_text = (
    "यह एक बड़ी खबर है जिसने प्रशासन को चौंका दिया है। पुरातत्व विभाग ने जांच शुरू की है। "
    "इस ऐतिहासिक खोज ने स्थानीय लोगों में उत्साह और जिज्ञासा पैदा कर दी है। अधिकारियों का कहना है कि "
    "इन मूर्तियों की ऐतिहासिकता की जांच की जा रही है।"
)
tts_audio = "output/tts_audio.mp3"
video_input = "output/right_section.mp4"

# Step 0: Generate TTS and left-side video
print("🎙️ Generating TTS and left text box video...")
generate_google_tts(input_text, tts_audio)
add_left_text_with_box(video_input, tts_audio, font_path, input_text, left_path)

# Step 1: Add silent audio to intro if needed
print("🔧 Ensuring intro has audio...")
subprocess.run([
    "ffmpeg", "-y",
    "-i", intro_path,
    "-f", "lavfi", "-t", "5",
    "-i", "anullsrc=channel_layout=mono:sample_rate=24000",
    "-c:v", "copy",
    "-shortest",
    "-c:a", "aac",
    intro_audio_path
], check=True)

# Step 2: Resize left_text video and fix audio to match intro
print("🔁 Adjusting left-side clip resolution and audio...")
subprocess.run([
    "ffmpeg", "-y",
    "-i", left_path,
    "-vf", "scale=1920:1080",
    "-ar", "24000",
    "-ac", "1",
    "-c:v", "libx264",
    "-c:a", "aac",
    left_fixed_path
], check=True)

# Step 3: Merge both videos with audio using concat
print("🎬 Merging intro and left-side video...")
subprocess.run([
    "ffmpeg", "-y",
    "-i", intro_audio_path,
    "-i", left_fixed_path,
    "-filter_complex",
    "[0:v:0][0:a:0][1:v:0][1:a:0]concat=n=2:v=1:a=1[outv][outa]",
    "-map", "[outv]", "-map", "[outa]",
    "-c:v", "libx264", "-c:a", "aac",
    merged_output
], check=True)

# Step 4: Add scrolling ticker to the bottom
print("📰 Adding ticker to merged video...")
add_scrolling_ticker(
    input_video_path=merged_output,
    output_video_path=final_with_ticker,
    font_path=font_path,
    ticker_text=ticker_text
)

print(f"✅ Final bulletin with ticker created: {final_with_ticker}")
