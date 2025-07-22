from PIL import Image

# Create transparent frame
frame = Image.new("RGBA", (1280, 720), (255, 0, 0, 100))  # Red with 100/255 opacity
frame.save("assets/test_frame.png")
