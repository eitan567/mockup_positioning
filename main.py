import cv2

# Load the image
image = cv2.imread(r'mockups\mockup4.png')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply edge detection
edges = cv2.Canny(gray, threshold1=100, threshold2=100)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours and draw black frames around each detected frame
min_area = image.shape[0] * image.shape[1] * 0.01  # at least 1% of the image area
for contour in contours:
    if cv2.contourArea(contour) > min_area:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 0), 2)


# Optionally show the image with black frames
# Instead of cv2.imshow
cv2.imwrite('output_with_black_frames.jpg', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
