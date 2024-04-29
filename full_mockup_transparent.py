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
image = cv2.imread(r'mockups\mockup4.png', cv2.IMREAD_UNCHANGED)

# # Convert to grayscale for edge detection
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Apply edge detection
# edges = cv2.Canny(gray, threshold1=150, threshold2=350)

# # Find contours
# contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
def non_maximum_suppression(boxes, overlapThresh):
    if len(boxes) == 0:
        return []

    # If the bounding boxes are integers, convert them to floats -- this
    # is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # Initialize the list of picked indexes
    pick = []

    # Grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2] + boxes[:,0]
    y2 = boxes[:,3] + boxes[:,1]

    # Compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # Keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # Grab the last index in the indexes list and add the index
        # value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # Find the largest (x, y) coordinates for the start of the
        # bounding box and the smallest (x, y) coordinates for the
        # end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # Compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # Compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # Delete all indexes from the index list that have overlap greater
        # than the provided threshold
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    # Return only the bounding boxes that were picked
    return boxes[pick].astype("int")

# Directory with images to insert into frames
image_directory = r'pictures'
image_files = [os.path.join(image_directory, f) for f in os.listdir(image_directory) if f.endswith(('jpg', 'png'))]

# # Convert to grayscale
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Apply GaussianBlur
# blurred = cv2.GaussianBlur(gray, (5,5), 0)

# # Canny edge detection
# edges = cv2.Canny(blurred, threshold1=100, threshold2=95)

# # Find contours
# contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# # Filter contours
# min_area = image.shape[0] * image.shape[1] * 0.01  # at least 1% of the image area
# frames = [(cv2.boundingRect(contour), contour) for contour in contours if cv2.contourArea(contour) > min_area]


# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply GaussianBlur
blurred = cv2.GaussianBlur(gray, (1,1), 0)

# Canny edge detection
edges = cv2.Canny(blurred, threshold1=100, threshold2=90)

# Find contours
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Here you define the non_maximum_suppression function
# [Define the non_maximum_suppression function here]

# Filter contours
min_area = image.shape[0] * image.shape[1] * 0.01  # at least 1% of the image area
contours_above_threshold = [contour for contour in contours if cv2.contourArea(contour) > min_area]

# Proceed with non-maximum suppression only if there are multiple contours above the threshold
if len(contours_above_threshold) > 1:
    boxes = np.array([cv2.boundingRect(contour) for contour in contours_above_threshold])
    filtered_boxes = non_maximum_suppression(boxes, 0.3)  # Adjust the threshold as needed
else:
    # If there's only one or no contour above the threshold, use what we have without suppression
    filtered_boxes = [cv2.boundingRect(contour) for contour in contours_above_threshold]

# Create the final list of frames using the filtered boxes
frames = []
for (x, y, w, h) in filtered_boxes:
    # Find the corresponding contour
    corresponding_contour = max(contours_above_threshold, key=lambda cont: cv2.contourArea(cont) if cv2.boundingRect(cont) == (x, y, w, h) else 0)
    frames.append(((x, y, w, h), corresponding_contour))

# Place each image into a frame
for frame, image_file in zip(frames, image_files):
    (x, y, w, h), contour = frame
    # Resize image to fit frame
    pil_image = resize_image_to_fit(image_file, w, h)
    # Convert PIL image to OpenCV format with alpha channel
    if pil_image.mode == 'RGBA':
        open_cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGBA2BGRA)
    else:
        open_cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    # Calculate position to center image in the frame
    start_x = x + (w - open_cv_image.shape[1]) // 2
    start_y = y + (h - open_cv_image.shape[0]) // 2
    # Create a mask for transparency from the alpha channel
    alpha_s = open_cv_image[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s
    # Blend the image into the frame based on the alpha mask
    for c in range(0, 3):
        image[start_y:start_y + open_cv_image.shape[0], start_x:start_x + open_cv_image.shape[1], c] = \
            (alpha_s * open_cv_image[:, :, c] +
             alpha_l * image[start_y:start_y + open_cv_image.shape[0], start_x:start_x + open_cv_image.shape[1], c])
    # if cv2.contourArea(contour) > min_area:
    #     x, y, w, h = cv2.boundingRect(contour)
    #     cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 0), 2)


# Optionally save the image with frames and inserted pictures
cv2.imwrite('mockup.png', image)  # Save as PNG to preserve transparency
cv2.waitKey(0)
cv2.destroyAllWindows()
