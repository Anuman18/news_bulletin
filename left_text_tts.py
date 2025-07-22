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
    import textwrap

    # Wrap text manually
    wrapped = textwrap.fill(input_text, width=20, break_long_words=False)
    ffmpeg_safe_text = wrapped.replace('\n', r'\n')

    # Box dimensions and padding
    box_width = 500
    box_height = 400
    box_x = 50
    box_y = 150
    text_x = box_x + 20
    text_y = box_y + 20

    # FFmpeg draw commands
    filter_complex = (
        f"[0:v]drawbox=x={box_x}:y={box_y}:w={box_width}:h={box_height}:"
        f"color=black@0.6:t=fill,"
        f"drawtext=fontfile='{font_path}':text='{ffmpeg_safe_text}':"
        f"fontcolor=white:fontsize=36:line_spacing=10:"
        f"x={text_x}:y={text_y}:box=0[v]"
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
    print(f"✅ Final video with left-side text/audio box: {output_path}")

    import textwrap

    # Step 1: Wrap text manually at ~25 characters per line (adjust as needed)
    wrapped_lines = textwrap.fill(input_text, width=25, break_long_words=False).replace('\n', r'\n')

    # Step 2: Define box dimensions (positioned left side)
    box_x = 50
    box_y = 100
    box_width = 700
    box_height = 300
    text_x = box_x + 30
    text_y = box_y + 30

    # Step 3: FFmpeg drawbox + drawtext
    filter_complex = (
        f"[0:v]drawbox=x={box_x}:y={box_y}:w={box_width}:h={box_height}:"
        f"color=black@0.6:t=fill,"
        f"drawtext=fontfile='{font_path}':"
        f"text='{wrapped_lines}':"
        f"fontcolor=white:fontsize=38:line_spacing=10:"
        f"x={text_x}:y={text_y}[v]"
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

    print("▶️ Adding properly wrapped text in box...")
    subprocess.run(cmd, check=True)
    print(f"✅ Done: {output_path}")
    # 📌 Manually wrap text to avoid overflow (you can make this dynamic later)
    wrapped_text = "यह एक बड़ी खबर है\nजिसने प्रशासन को\nचौंका दिया है।"

    # Position and size of the box
    box_x = 50
    box_y = 100
    box_width = 600
    box_height = 220
    text_x = box_x + 20
    text_y = box_y + 20

    # FFmpeg filter
    filter_complex = (
        f"[0:v]drawbox=x={box_x}:y={box_y}:w={box_width}:h={box_height}:"
        f"color=black@0.5:t=fill,"
        f"drawtext=fontfile='{font_path}':"
        f"text='{wrapped_text}':"
        f"fontcolor=white:fontsize=34:line_spacing=10:"
        f"x={text_x}:y={text_y}[v]"
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

    print("▶️ Adding left-side box with wrapped text...")
    subprocess.run(cmd, check=True)
    print(f"✅ Final video with boxed text/audio: {output_path}")

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
    input_text = (
    "यह एक बड़ी खबर है जिसने बड़ी खबर है जिसने\n"
    "प्रशासन को चौंका दिया है।बड़ी खबर है जिसने\n"
    "पुरातत्व विभाग ने जांचबड़ी खबर है जिसने\n"
    "शुरू की है।बड़ी खबर है जिसने \n"
    "प्रशासन को चौंका दिया है।बड़ी खबर है जिसने\n"
    "प्रशासन को चौंका दिया है।बड़ी खबर है जिसने\n"
)
    tts_audio = "output/tts_audio.mp3"
    video_input = "output/right_section.mp4"
    final_output = "output/left_text_with_tts.mp4"
    font_path = "assets/hindi-bold.ttf"

    generate_google_tts(input_text, tts_audio)
    add_left_text_box(video_input, tts_audio, font_path, input_text, final_output)
