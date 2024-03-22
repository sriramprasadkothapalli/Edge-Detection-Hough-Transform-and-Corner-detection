## Edge Detection, Hough-Transform and Corner-detection

- Designed a video processing pipeline to find the 4 corners of the paper using Edge Detection, Hough Transform and
Corner Detection. Overlay the boundary(edges) of the paper and highlight the four corners of the paper and then verified the existence of corners using Harris corner detection
# Libraries
Import the following libraries
```
import os
from google.colab import drive
from google.colab.patches import cv2_imshow
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
```
# Pipeline 

- Reads the video frame by frame and checks if it's successfully read.

- Uses a function (is_blurry) to analyze the frame and determine if it's too blurry for further processing.

- If the frame isn't blurry, it uses Canny edge detection to highlight edges and details in the image.

- Once edges are identified, another function (detect_lines) uses Hough Lines P to find lines present in the edge map.

- Analyze the detected lines, potentially identifying dominant lines or calculating their intersections.

- Draws blue lines on the frame to show all detected lines, and adds red circles at the intersection points (if found).

- Writes the modified frame (with lines and intersections) to a new video file

- Prints the total number of blurry frames encountered and the number of non-blurry frames that were processed

  # Results
- Applying Canny Edge Detection
  
  ![canny](https://github.com/sriramprasadkothapalli/Edge-Detection-Hough-Transform-and-Corner-detection/assets/143056659/746c315c-4095-4fd4-851c-f4248f647955)
- Line and corner detection
  
  ![line detection](https://github.com/sriramprasadkothapalli/Edge-Detection-Hough-Transform-and-Corner-detection/assets/143056659/e9a8e02a-d981-4b6d-a650-522d615fca81)
- Verifying using Harris corner detection
  
  ![with harris](https://github.com/sriramprasadkothapalli/Edge-Detection-Hough-Transform-and-Corner-detection/assets/143056659/e41b5383-b062-4d07-9f93-159088aff596)

