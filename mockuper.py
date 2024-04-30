from PIL import Image
import numpy as np
import os

def blend_multiply(image1, image2):
    # Convert images to 'float32' to prevent data type overflow
    arr1 = np.array(image1, dtype='float32')
    arr2 = np.array(image2, dtype='float32')
    
    # Multiply the images
    multiplied = arr1 * arr2 / 255  # Normalize the result since values can range from 0 to 255
    
    # Convert back to 'uint8'
    multiplied = multiplied.astype('uint8')
    
    # Convert array back to Image
    return Image.fromarray(multiplied)

def merge_images(mockup_path, art_path, position, size):
    with Image.open(mockup_path) as mockup_image, Image.open(art_path) as art_image:
        art_image = art_image.resize(size, Image.ANTIALIAS)
        art_image = art_image.convert("RGBA")  # Ensure image has an alpha channel
        mockup_image = mockup_image.convert("RGBA")
        
        # Apply the multiply blend mode
        x, y = position
        mockup_region = mockup_image.crop((x, y, x + size[0], y + size[1]))
        blended_region = blend_multiply(mockup_region, art_image)
        
        # Paste the blended region back
        mockup_image.paste(blended_region, position, blended_region)
        
        return mockup_image

# Example usage
mockup_path = os.environ.get("MOCKUP_FILE")
art_path = os.environ.get("ART_FILE")
position = (450, 600)  # x, y coordinates on the mockup
size = (1100, 1100)    # width, height of the art image

result_image = merge_images(mockup_path, art_path, position, size)
result_image.save('mockup1.png')
# result_image.show()
