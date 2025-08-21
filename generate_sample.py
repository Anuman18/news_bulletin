# pip install google-cloud-texttospeech pydub
# Make sure FFmpeg is installed and on PATH

from google.cloud import texttospeech
from pydub import AudioSegment
import re, io, os

# 1) Auth
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key.json"  # your service account

VOICE_NAME = "hi-IN-Chirp3-HD-Despina"
LANG_CODE = "hi-IN"

# --- Pause mapping ---
# Tune this to taste. Example: every 2 dots ~ 1 second, capped at 4s.
def dots_to_ms(dots: str) -> int:
    n = len(dots)
    seconds = min(max(n // 2, 1), 4)   # "..." => 1s, "....." => 2s, "......." => 3s, cap 4s
    return seconds * 1000

PAUSE_PATTERN = re.compile(r"\.{3,}")  # 3 or more dots trigger a pause

def parse_segments(text: str):
    """
    Split text into a sequence of ('text', chunk) and ('pause', ms) items
    based on runs of 3+ dots.
    """
    parts = re.split(f"({PAUSE_PATTERN.pattern})", text)
    for part in parts:
        if not part:
            continue
        if PAUSE_PATTERN.fullmatch(part):
            yield ("pause", dots_to_ms(part))
        else:
            yield ("text", part)

def chunk_long_text(s: str, max_len: int = 3800):
    """
    Split long text into <= max_len chunks on sentence-ish boundaries.
    This avoids API length issues on very long inputs.
    """
    s = s.strip()
    if len(s) <= max_len:
        return [s]
    chunks, buf = [], []
    length = 0
    for piece in re.split(r"(\s+)", s):  # keep whitespace
        if length + len(piece) > max_len and buf:
            chunks.append("".join(buf).strip())
            buf, length = [piece], len(piece)
        else:
            buf.append(piece)
            length += len(piece)
    if buf:
        chunks.append("".join(buf).strip())
    return chunks

def synthesize_text_segment(client, text_chunk: str) -> AudioSegment:
    """
    Synthesize a single text chunk and return as a pydub AudioSegment.
    We must use 'text=' (not SSML) because Chirp3-HD doesn't accept SSML.
    """
    synthesis_input = texttospeech.SynthesisInput(text=text_chunk)

    voice_params = texttospeech.VoiceSelectionParams(
        language_code=LANG_CODE,
        name=VOICE_NAME,
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL,  # ignored by named voice, still safe
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=0.85,
    )

    resp = client.synthesize_speech(
        input=synthesis_input,
        voice=voice_params,
        audio_config=audio_config
    )

    return AudioSegment.from_file(io.BytesIO(resp.audio_content), format="mp3")

def synthesize_with_silences(raw_text: str, out_path: str = "output_despina.mp3"):
    client = texttospeech.TextToSpeechClient()

    final = AudioSegment.silent(duration=0)
    for kind, payload in parse_segments(raw_text):
        if kind == "text":
            # Break large text parts further if needed
            chunks = chunk_long_text(payload)
            for ch in chunks:
                if ch.strip():
                    seg = synthesize_text_segment(client, ch)
                    final += seg
        else:  # pause
            final += AudioSegment.silent(duration=payload)

    final.export(out_path, format="mp3")
    print(f"✅ Saved: {out_path}")

if __name__ == "__main__":
    sample_text = (
        "Episode 4: The Kindness Compass... Hello, friends! A compass helps a traveler find their direction. Today, we're going to talk about an imaginary compass inside you that always points to being kind! It's your Kindness Compass... Kindness isn't just about being 'nice.' It's about actively choosing to be helpful, caring, and understanding towards others. Being known for your kindness is one of the best personal brands you can have... Your Kindness Compass can point in many directions. It can point to Kindness to Friends, like sharing your lunch, or Kindness to Family, like helping with chores without being asked... It also points to Kindness to Nature, like watering the plants, and most importantly, Kindness to Yourself, which means being kind to yourself when you make a mistake... When you are kind to one person, it can inspire them to be kind to someone else. This is called the ripple effect. Your one small act can create a huge wave of positivity... Being kind also helps you build trust. When you are consistently kind, people know they can come to you for help. It's a very strong part of your personal brand that people will remember you for... Now, for your mission: The Kindness Compass Challenge! Pause the video for just one minute... Draw a compass in your notebook. In each direction, write down one thing you can do this week to be kind to your friends, family, nature, and yourself. You'll soon see a whole new world of possibilities... You did it! That's it for today's episode. Thank you for joining us! We'll be back soon with more episodes of Self-Branding. Keep being curious!..."
    )
    synthesize_with_silences(sample_text, out_path="sample_hiIN_despina_paused.mp3")
