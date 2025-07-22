from google.cloud import texttospeech
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key.json"

client = texttospeech.TextToSpeechClient()

synthesis_input = texttospeech.SynthesisInput(text="यह एक परीक्षण वाक्य है।")

voice = texttospeech.VoiceSelectionParams(
    language_code="hi-IN",
    name="hi-IN-Wavenet-D"
)

audio_config = texttospeech.AudioConfig(
    audio_encoding=texttospeech.AudioEncoding.MP3
)

response = client.synthesize_speech(
    input=synthesis_input,
    voice=voice,
    audio_config=audio_config
)

with open("output/test_tts.mp3", "wb") as out:
    out.write(response.audio_content)

print("✅ Audio generated: output/test_tts.mp3")
