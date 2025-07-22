# main.py
import os
import subprocess

# Sequential steps for generating final bulletin
def run_script(script_name):
    print(f"\n▶️ Running {script_name}...")
    result = subprocess.run(["python", script_name], capture_output=True, text=True)
    if result.returncode == 0:
        print(result.stdout)
    else:
        print(f"❌ Error in {script_name}:\n{result.stderr}")
        exit(1)

def main():
    print("🚀 Generating Full News Bulletin...")

    # 1. Add intro (optional step, remove if not used)
    run_script("base_background.py")

    # 2. Overlay image/video on the right side
    run_script("right_section.py")

    # 3. Generate TTS audio and text overlay on left side
    run_script("left_text_tts.py")

    # 4. Add scrolling ticker
    run_script("scrolling_ticker.py")

    # 5. Merge all parts into full bulletin
    run_script("merge_all.py")

    print("✅ All done! Final output: output/full_bulletin.mp4")

if __name__ == "__main__":
    main()
