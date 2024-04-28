import cv2
import os
from PIL import Image
import numpy as np

def resize_image_to_fit(image_path, width, height):
    img = Image.open(image_path)
    img_aspect = img.width / img.height
    frame_aspect = width / height

    if img_aspect > frame_aspect:
        # Image is wider than the frame, fit to width
        new_width = width
        new_height = int(new_width / img_aspect)
    else:
        # Image is taller or equal to frame, fit to height
        new_height = height
        new_width = int(new_height * img_aspect)

    resized_img = img.resize((new_width, new_height), Image.LANCZOS)
    return resized_img

# Load the base image (mockup)
image = cv2.imread(r'mockups\mockup1.png')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply edge detection
edges = cv2.Canny(gray, threshold1=100, threshold2=100)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours
min_area = image.shape[0] * image.shape[1] * 0.01  # at least 1% of the image area
frames = [(cv2.boundingRect(contour), contour) for contour in contours if cv2.contourArea(contour) > min_area]

# Directory with images to insert into frames
image_directory = r'pictures'
image_files = [os.path.join(image_directory, f) for f in os.listdir(image_directory) if f.endswith(('jpg', 'png'))]

# Place each image into a frame
for frame, image_file in zip(frames, image_files):
    (x, y, w, h), contour = frame
    # Resize image to fit frame
    pil_image = resize_image_to_fit(image_file, w, h)
    # Convert PIL image to OpenCV format
    open_cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    # Calculate position to center image in the frame
    start_x = x + (w - open_cv_image.shape[1]) // 2
    start_y = y + (h - open_cv_image.shape[0]) // 2
    # Place image in the frame
    image[start_y:start_y + open_cv_image.shape[0], start_x:start_x + open_cv_image.shape[1]] = open_cv_image

# Optionally save the image with frames and inserted pictures
cv2.imwrite('mockup.jpg', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
