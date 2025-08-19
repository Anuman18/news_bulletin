from google.cloud import texttospeech
import os

# Set your Google Cloud service account key
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key.json"  # Replace with your key file

def generate_sample(language_code="hi-IN", voice_name="hi-IN-Chirp3-HD-Despina", gender="NEUTRAL"):
    client = texttospeech.TextToSpeechClient()

    # Example sample text
    sample_text = {
        "hi-IN": "Hello, friends! What’s the first thing you think of when you hear the word 'superpower'? Flying? Super strength? While those are cool, we all have real-life superpowers inside us!" "A personal superpower isn't about being the best at everything. It’s about what you do well and what makes you feel proud. Maybe your superpower is being a super kind friend, a fantastic artist, or a math whiz!" "Examples of personal superpowers are the Creative Genius, who loves to draw; the Problem Solver, who is great at puzzles; and the Caring Captain, who makes sure everyone feels included." "The first step to building your personal brand is to find your own superpower! Think about what you enjoy and what you're good at. You can be a mix of many things!" "Knowing your superpower is the first step. The next is to use it! When you use your superpower, you not only feel happy and confident but you also make the world around you a better place." "When people think of you, they’ll think of your amazing superpower! Just like we think of Sachin Tendulkar and remember his 'Dedication' superpower. This is the foundation of your personal brand!" "Now, for your mission: The Superpower Chart! Pause the video for just one minute." "On a piece of paper, create your own superpower chart. Write down what you enjoy and what you're good at. You'll soon see a whole new world of possibilities!" "You did it! That's it for today's episode. Thank you for joining us! We'll be back soon with more episodes of Self-Branding. Keep being curious!",
    }.get(language_code, "This is a sample voice from Google Text to Speech.")

    synthesis_input = texttospeech.SynthesisInput(text=sample_text)

    # Select the voice
    voice_params = texttospeech.VoiceSelectionParams(
        language_code=language_code,
        name=voice_name,
        ssml_gender=texttospeech.SsmlVoiceGender[gender]
    )

    # Audio config
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    # Generate speech
    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice_params,
        audio_config=audio_config
    )

    # Save to file
    output_file = f"sample_{language_code}.mp3"
    with open(output_file, "wb") as out:
        out.write(response.audio_content)
        print(f"✅ Sample voice saved as {output_file}")

if __name__ == "__main__":
    # Example: change language_code to try different voices
    # language_code examples: "en-US", "hi-IN", "fr-FR"
    generate_sample(language_code="hi-IN", gender="FEMALE")

