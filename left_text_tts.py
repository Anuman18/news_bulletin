import os
import subprocess
from pydub import AudioSegment
from google.cloud import texttospeech

# Google TTS setup
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key.json"

def generate_google_tts(text, out_path, lang="hi-IN", voice="hi-IN-Wavenet-D"):
    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=text)

    voice_params = texttospeech.VoiceSelectionParams(
        language_code=lang,
        name=voice
    )
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

    response = client.synthesize_speech(
        input=synthesis_input, voice=voice_params, audio_config=audio_config
    )

    with open(out_path, "wb") as out:
        out.write(response.audio_content)
    print(f"✅ TTS audio saved: {out_path}")

def add_left_text_box(video_input, tts_audio, font_path, input_text, output_path):
    # Sanitize input text
    safe_text = input_text.replace("'", "’").replace("\n", " ")

    # Width and height of the textbox area (adjust as needed)
    box_width = 600
    box_height = 300
    box_x = 50
    box_y = 100

    filter_complex = (
        f"[0:v]drawbox=x={box_x}:y={box_y}:w={box_width}:h={box_height}:color=black@0.5:t=fill,"
        f"drawtext=fontfile='{font_path}':text='{safe_text}':"
        f"fontcolor=white:fontsize=34:box=0:"
        f"x={box_x + 20}:y={box_y + 20}:"
        f"line_spacing=10[v]"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", video_input,
        "-i", tts_audio,
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-map", "1:a",
        "-c:v", "libx264",
        "-c:a", "aac",
        "-pix_fmt", "yuv420p",
        "-shortest",
        output_path
    ]

    subprocess.run(cmd, check=True)
    print(f"✅ Final video with left-side text/audio in box: {output_path}")

    filter_complex = (
        f"[0:v]drawtext=fontfile='{font_path}':"
        f"text='{input_text}':fontcolor=white:fontsize=36:"
        f"x=40:y=100:line_spacing=10:box=1:boxcolor=black@0.5:boxborderw=10[v]"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", video_input,
        "-i", tts_audio,
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-map", "1:a",
        "-c:v", "libx264",
        "-c:a", "aac",
        "-pix_fmt", "yuv420p",
        "-shortest",
        output_path
    ]

    subprocess.run(cmd, check=True)
    print(f"✅ Final video with left-side text/audio: {output_path}")


# 🧪 Example use
if __name__ == "__main__":
    input_text = "यह एक बड़ी खबर है जिसने प्रशासन को चौंका दिया है।"
    tts_audio = "output/tts_audio.mp3"
    video_input = "output/right_section.mp4"
    final_output = "output/left_text_with_tts.mp4"
    font_path = "assets/hindi-bold.ttf"

    generate_google_tts(input_text, tts_audio)
    add_left_text_box(video_input, tts_audio, font_path, input_text, final_output)
