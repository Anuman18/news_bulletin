import os
import subprocess
import textwrap
from google.cloud import texttospeech

# Google TTS setup
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key.json"

def generate_google_tts(text, out_path, lang="hi-IN", voice="hi-IN-Wavenet-D"):
    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice_params = texttospeech.VoiceSelectionParams(
        language_code=lang, name=voice
    )
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice_params, audio_config=audio_config
    )
    with open(out_path, "wb") as out:
        out.write(response.audio_content)
    print(f"✅ TTS audio saved: {out_path}")

def add_left_text_with_box(video_input, tts_audio, font_path, input_text, output_path):
    import textwrap
    import os
    import subprocess
    
    # Calculate optimal text wrapping based on text length
    text_length = len(input_text)
    if text_length < 150:
        wrap_width = 32
        font_size = 36
    elif text_length < 250:
        wrap_width = 35
        font_size = 32
    elif text_length < 350:
        wrap_width = 38
        font_size = 28
    else:
        wrap_width = 42
        font_size = 26
    
    # Neatly wrap text
    wrapped = textwrap.fill(input_text, width=wrap_width, break_long_words=False)
    
    # Write to temp file
    temp_text_file = "temp_text.txt"
    with open(temp_text_file, "w", encoding="utf-8") as f:
        f.write(wrapped)
    
    # Video resolution
    WIDTH = 1920
    HEIGHT = 1080
    
    # Text positioning (top-left, avoiding image area)
    line_spacing = 8
    text_x = 60
    text_y = 120  # Start from top with some margin
    
    # Create the filter complex with just text (no box)
    filter_complex = (
        f"[0:v]drawtext=fontfile='{font_path}':textfile='{temp_text_file}':"
        f"fontcolor=white:fontsize={font_size}:line_spacing={line_spacing}:"
        f"x={text_x}:y={text_y}:shadowcolor=black:shadowx=2:shadowy=2[v]"
    )
    
    cmd = [
        "ffmpeg", "-y",
        "-i", video_input,
        "-i", tts_audio,
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-map", "1:a",
        "-c:v", "libx264", "-c:a", "aac",
        "-pix_fmt", "yuv420p",
        "-shortest", output_path
    ]
    
    print("🎬 Rendering video with text box...")
    subprocess.run(cmd, check=True)
    print(f"✅ Final video saved to: {output_path}")
    
    # Clean up temp file
    os.remove(temp_text_file)

# Example use
if __name__ == "__main__":
    input_text = (
        "'सुनहरी सखी' महिला स्व-सहायता समूह ने हर्बल साबुन और तेल बेचकर प्रति माह "
        "₹50,000 की कमाई शुरू की है। यह पहल महिला आत्मनिर्भरता की दिशा में एक बड़ा कदम है।"
    )
    
    tts_audio = "output/tts_audio.mp3"
    video_input = "output/right_section.mp4"
    final_output = "output/left_text_with_tts.mp4"
    font_path = "assets/hindi-bold.ttf"
    
    generate_google_tts(input_text, tts_audio)
    add_left_text_with_box(video_input, tts_audio, font_path, input_text, final_output)