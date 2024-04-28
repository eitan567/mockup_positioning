from PIL import Image

# Load an image
image = Image.open(r'pictures\1.png')

# Define the new size
new_size = 5000  # width, height in pixels

# Calculate aspect ratio preservation
aspect_ratio = image.width / image.height
new_height = int(new_size / aspect_ratio)
resized_image = image.resize((new_size, new_height), Image.LANCZOS)

# Image.NEAREST, Image.BOX, Image.BILINEAR, Image.HAMMING, Image.BICUBIC, and Image.LANCZOS

# Save the resized image
resized_image.save(f'pillow_resized_image_{new_size}X{new_size}.png')

# Optionally display the resized image
resized_image.show()
