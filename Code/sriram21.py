import os
from google.colab import drive
from google.colab.patches import cv2_imshow
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Mount your Google Drive for file access
drive.mount('/content/drive/')

# Define the path to your project folder
path_to_folder = "ENPM673"
%cd /content/drive/My\ Drive/{path_to_folder}

# Specify the location for storing output frames
output_folder_path = os.path.join('/content/drive/My Drive/ENPM673/', 'frames_store_project2')
output_video_path = os.path.join('/content/drive/My Drive/ENPM673/', 'frames_video_project2.mp4')
# Create the output folder if it doesn't already exist
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

def is_blurry(image, threshold=80):
    """Determines blurriness using filter2D and image variance."""
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Sharpening Kernel (adjust values as needed)
    kernel = np.array([[0, 1, 0],
                       [1,  -4, 1],
                       [0, 1, 0]])

    # Apply the sharpening filter
    sharpened = cv.filter2D(gray, -1, kernel)

    # Calculate the variance
    variance = sharpened.var()

    return variance < threshold

def Canny_Edge_Detection(image, low_threshold, high_threshold):
  gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
  edges = cv.Canny(gray, low_threshold, high_threshold)
  return edges

def detect_lines(canny_edges, min_line_length, max_line_gap):
    """Applies Hough Line Transform and filters lines."""
    lines = cv.HoughLinesP(canny_edges, rho=1, theta=np.pi/180,
                             threshold=65, minLineLength=min_line_length,
                             maxLineGap=max_line_gap)
    return lines

def find_intersections(lines):
    """Calculates intersections between lines."""
    intersections = []
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            line1 = lines[i][0]
            line2 = lines[j][0]

            # Obtain line equations (in the form y = mx + c)
            x1, y1, x2, y2 = line1
            m1 = (y2 - y1) / (x2 - x1)
            c1 = y1 - m1 * x1

            x3, y3, x4, y4 = line2
            m2 = (y4 - y3) / (x4 - x3)
            c2 = y3 - m2 * x3

            # Calculate intersection point
            if (m1 - m2) != 0:
                x = (c2 - c1) / (m1 - m2)
                y = m1 * x + c1
                intersections.append((int(x), int(y)))
            # else: Lines are parallel
    return intersections


def find_dominant_lines(lines, max_lines=4):
    """Keeps only a specified number of dominant lines (based on length)."""
    def line_length(line):
        x1, y1, x2, y2 = line[0]
        return ((x2 - x1)**2 + (y2 - y1)**2) ** 0.5

    sorted_lines = sorted(lines, key=line_length, reverse=True)
    return sorted_lines[:max_lines]


def harris_corner_detection(image, threshold=0.01):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    # Result is dilated for marking the corners
    dst = cv.dilate(dst, None)
    # Threshold for an optimal value, marking the corners in red
    image[dst > threshold * dst.max()] = [0, 255, 0]
    return image


# Open the video file
cap = cv.VideoCapture('proj2_v2.mp4')

# Determine the video's width and height to use with VideoWriter
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

output = cv.VideoWriter(output_video_path, cv.VideoWriter_fourcc(*'mp4v'), 10, (frame_width, frame_height))

blurry_count = 0
total_frames = 0
while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:  # Check if a frame was successfully read
        break
    total_frames += 1  # Increment total frame count
    frame_number = cap.get(cv.CAP_PROP_POS_FRAMES)
    #frame = gaussian_blur(frame, kernel_size=3)
    if  is_blurry(frame):

        blurry_count += 1
    else:
      low_threshold = 750
      high_threshold = 850
      canny_edges = Canny_Edge_Detection(frame, low_threshold, high_threshold)
      #cv2_imshow(canny_edges)
      lines = detect_lines(canny_edges, min_line_length=80, max_line_gap=3)

        # Find dominant lines and intersections
      dominant_lines = find_dominant_lines(lines)
      intersections = find_intersections(dominant_lines)
      if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)

        # Draw dominant lines (blue, thicker)
      if dominant_lines:
            for line in dominant_lines:
                x1, y1, x2, y2 = line[0]
                cv.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)

        # Draw intersections (red circles)
      if intersections:
            for point in intersections:
                x, y = point
                cv.circle(frame, (x, y), radius=5, color=(0, 0, 255), thickness=-1)
      filename = f'frame_{int(frame_number)}_edges.jpg'
      filepath = os.path.join(output_folder_path, filename)
     # frame = harris_corner_detection(frame)
      output.write(frame)
     # cv2_imshow(frame)

 # Save the frame

    white_condition = np.all(frame > 200, axis=2)
    white_indices = np.where(white_condition)

    background = np.zeros_like(frame)

    frame[~white_condition] = background[~white_condition]




         # Exit the loop if 'q' is pressed
    if cv.waitKey(10) & 0xFF == ord('q'):
        break
# Release resources when finished with the video
output.release()
cap.release()
cv.destroyAllWindows()


print(f"Number of blurry images: {blurry_count}")
print(f"Number of frames extracted: {total_frames - blurry_count}")
