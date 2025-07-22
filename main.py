import os
import subprocess
import textwrap
from pydub import AudioSegment
from google.cloud import texttospeech

# Set your Google credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key.json"


def generate_tts(text, output_audio_path, lang="hi-IN", voice_name="hi-IN-Wavenet-D"):
    client = texttospeech.TextToSpeechClient()

    synthesis_input = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
        language_code=lang,
        name=voice_name
    )

    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

    response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)

    with open(output_audio_path, "wb") as out:
        out.write(response.audio_content)
        print(f"✅ Google TTS audio saved: {output_audio_path}")


def generate_news_ticker_clip(image_path, audio_path, frame_path, font_path, output_path, text):
    # Get audio duration
    audio = AudioSegment.from_file(audio_path)
    duration = audio.duration_seconds

    # Sanitize scrolling text
    text_safe = text.replace("'", "’").replace("\n", " ")

    filter_complex = (
        f"[0:v]scale=1280:720,format=rgba[bg];"
        f"[2:v]scale=1280:720[frame];"
        f"[bg][frame]overlay=0:0[framed];"
        f"[framed]drawtext=fontfile='{font_path}':"
        f"text='{text_safe}':"
        f"fontcolor=white:fontsize=38:"
        f"x=w-mod(t*120\\,w+tw):y=h-55:"
        f"shadowcolor=black:shadowx=2:shadowy=2[v]"
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-loop", "1",
        "-t", str(duration),
        "-i", image_path,
        "-i", audio_path,
        "-i", frame_path,
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-map", "1:a",
        "-c:v", "libx264",
        "-c:a", "aac",
        "-pix_fmt", "yuv420p",
        "-shortest",
        output_path
    ]

    print(f"▶️ Generating: {output_path}")
    subprocess.run(cmd, check=True)
    print(f"✅ Final video saved: {output_path}")


# 🧪 Run Example
if __name__ == "__main__":
    input_text = "संभल में मिली सैकड़ों मूर्तियाँ प्रशासन को चौंका रही हैं। यह खोज भारत की सांस्कृतिक विरासत के लिए बेहद अहम मानी जा रही है।"

    # 1. Generate audio from text
    tts_audio_path = "assets/clips/audio_tts.mp3"
    generate_tts(input_text, tts_audio_path)

    # 2. Create news ticker video
    generate_news_ticker_clip(
        image_path="assets/clips/img1.jpg",
        audio_path=tts_audio_path,
        frame_path="assets/frame.png",
        font_path="assets/hindi-bold.ttf",
        output_path="output/news_ticker_google_tts.mp4",
        text=input_text
    )
