import os
import subprocess
from moviepy.editor import VideoFileClip, ImageClip, concatenate_videoclips
from PIL import Image
from mutagen.mp3 import MP3

# Output folder
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_audio_duration(audio_path):
    audio = MP3(audio_path)
    return audio.info.length

def prepare_right_clip(media_path, duration, output_path):
    ext = os.path.splitext(media_path)[1].lower()
    
    if ext in ['.jpg', '.jpeg', '.png']:
        # For image, display it for the full audio duration
        img = Image.open(media_path)
        clip = ImageClip(media_path).set_duration(duration).resize(height=720)
    elif ext in ['.mp4', '.mov', '.avi']:
        # For video, loop it until it matches the audio duration
        original = VideoFileClip(media_path)
        loops = int(duration // original.duration) + 1
        repeated_clips = [original] * loops
        final = concatenate_videoclips(repeated_clips).subclip(0, duration)
        clip = final.resize(height=720)
    else:
        raise ValueError("Unsupported file format. Use image or video.")

    clip.write_videofile(output_path, fps=25, codec='libx264')
    clip.close()

def merge_audio_with_video(video_path, audio_path, output_path):
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest", output_path
    ]
    subprocess.run(cmd, check=True)

# Example usage
media_path = "assets/sample_video.mp4"     # or use an image like "assets/sample_image.jpg"
audio_path = "output/tts_audio.mp3"
temp_video = os.path.join(OUTPUT_DIR, "right_temp.mp4")
final_output = os.path.join(OUTPUT_DIR, "right_section.mp4")

# Generate final video
audio_duration = get_audio_duration(audio_path)
prepare_right_clip(media_path, audio_duration, temp_video)
merge_audio_with_video(temp_video, audio_path, final_output)
