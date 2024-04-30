from PIL import Image
import numpy
from blend_modes import soft_light
from dotenv import load_dotenv
import os

mockup_path = os.environ.get("MOCKUP_FILE")
art_path = os.environ.get("ART_FILE")

# Import background image
background_img_raw = Image.open(mockup_path)  # RGBA image
background_img = numpy.array(background_img_raw)  # Inputs to blend_modes need to be numpy arrays.
background_img_float = background_img.astype(float)  # Inputs to blend_modes need to be floats.

# Import foreground image
foreground_img_raw = Image.open(art_path)  # RGBA image
foreground_img = numpy.array(foreground_img_raw)  # Inputs to blend_modes need to be numpy arrays.
foreground_img_float = foreground_img.astype(float)  # Inputs to blend_modes need to be floats.

def add_alpha_channel(image):
    height, width = image.shape[:2]
    # Create an alpha channel with full opacity for every pixel
    alpha_channel = np.ones((height, width, 1), dtype=image.dtype) * 255
    return np.concatenate((image, alpha_channel), axis=-1)
    
background_img_float = add_alpha_channel(cv2.imread(mockup_path, -1).astype(float))

# Blend images
opacity = 0.7  # The opacity of the foreground that is blended onto the background is 70 %.
blended_img_float = soft_light(background_img_float, foreground_img_float, opacity)

# Convert blended image back into PIL image
blended_img = numpy.uint8(blended_img_float)  # Image needs to be converted back to uint8 type for PIL handling.
blended_img_raw = Image.fromarray(blended_img)  # Note that alpha channels are displayed in black by PIL by default.
                                                # This behavior is difficult to change (although possible).
                                                # If you have alpha channels in your images, then you should give
                                                # OpenCV a try.

# Display blended image
blended_img_raw.show()