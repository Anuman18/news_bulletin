import os
import subprocess
import textwrap
from google.cloud import texttospeech

# Google TTS setup
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key.json"

def create_ass_subtitle(text, ass_path="text.ass"):
    import textwrap
    wrapped_lines = textwrap.wrap(text, width=45, break_long_words=False)
    ass_events = ""
    start_time = 0
    duration = 3  # seconds per line

    for i, line in enumerate(wrapped_lines):
        start = f"0:00:{start_time:02d}.00"
        end = f"0:00:{start_time + duration:02d}.00"
        ass_events += f"Dialogue: 0,{start},{end},BoxedText,,0,0,0,,{line}\\N\n"
        start_time += duration

    ass_content = f"""[Script Info]
Title: Left Side Text
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Alignment, MarginL, MarginR, MarginV, Encoding
Style: BoxedText,Noto Sans,36,&H00FFFFFF,&H80000000,-1,0,0,0,100,100,0,7,100,50,100,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
{ass_events}"""

    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(ass_content)
    print(f"✅ ASS file created: {ass_path}")


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

def add_left_text_box(video_input, tts_audio, input_text, output_path):
    # Create subtitle file
    create_ass_subtitle(input_text, "text.ass")

    # FFmpeg command to overlay ASS subtitles
    cmd = [
        "ffmpeg", "-y",
        "-i", video_input,
        "-i", tts_audio,
        "-vf", "subtitles=text.ass",
        "-map", "0:v", "-map", "1:a",
        "-c:v", "libx264", "-c:a", "aac",
        "-pix_fmt", "yuv420p",
        "-shortest",
        output_path
    ]

    print("🎬 Adding ASS subtitle overlay...")
    subprocess.run(cmd, check=True)
    print(f"✅ Final video generated: {output_path}")

# 🧪 Example use
if __name__ == "__main__":
    input_text = (
        "यह एक बड़ी खबर है जिसने प्रशासन को चौंका दिया है। "
        "पुरातत्व विभाग ने इस मामले की जांच शुरू की है और यह खोज ऐतिहासिक महत्व की हो सकती है। "
        "अधिकारियों का कहना है कि मूर्तियों की उत्पत्ति का पता लगाने के लिए गहन अध्ययन किया जाएगा। "
        "स्थानीय लोग इस खोज को लेकर बेहद उत्साहित हैं।"
    )
    tts_audio = "output/tts_audio.mp3"
    video_input = "output/right_section.mp4"
    final_output = "output/left_text_with_tts.mp4"

    generate_google_tts(input_text, tts_audio)
    add_left_text_box(video_input, tts_audio, input_text, final_output)
