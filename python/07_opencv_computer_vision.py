"""
OpenCV — Computer Vision

Description:
OpenCV is a comprehensive library for image processing and computer vision. It is widely used for tasks such as image transformation, feature detection, and real-time video analysis.

Key Functionalities:
- Image and video reading/writing
- Image transformation (resize, crop, rotate)
- Feature detection (edges, faces, objects)
- Real-time video processing
- Integration with deep learning models

Sample Code:
"""

import cv2

img = cv2.imread('image.jpg')  # Make sure 'image.jpg' exists!
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('gray_image.jpg', gray)
print("Converted image to grayscale and saved as 'gray_image.jpg'.")