# People Detection with YOLOv5

This project uses YOLOv5 to detect people in an image. It highlights detected people with bounding boxes, draws a horizontal line for reference, counts the number of people, and displays the current timestamp.

## Features
- Detects only people in an image using YOLOv5.
- Draws bounding boxes around detected people.
- Displays a horizontal reference line.
- Shows the total people count on the image.
- Adds the current timestamp to the image.
- Saves the output image as `people_detected_image.jpg`.

## Requirements
- Python 3.x
- PyTorch
- OpenCV

Install dependencies with:

```bash
pip install torch torchvision opencv-python
