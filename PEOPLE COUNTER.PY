import torch
import cv2
import datetime
from pathlib import Path

# Load YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.4  # confidence threshold

# Image input
image_path = input("Enter image path: ")
frame = cv2.imread(image_path)
if frame is None:
    print("Image not found!")
    exit()

height, width = frame.shape[:2]
line_y = height // 2
font = cv2.FONT_HERSHEY_SIMPLEX
people_count = 0

# YOLO inference
results = model(frame)
detections = results.xyxy[0].cpu().numpy()

for det in detections:
    x1, y1, x2, y2, conf, cls = det
    if int(cls) == 0:  # only people
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        people_count += 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

# Draw line and info
cv2.line(frame, (0, line_y), (width, line_y), (0, 255, 0), 2)
cv2.putText(frame, f'People Count: {people_count}', (10, 30), font, 1, (0, 0, 255), 2)
cv2.putText(frame, str(datetime.datetime.now()), (10, height - 10), font, 0.5, (0, 0, 255), 1)

# Show and save
cv2.imshow("People Detection", frame)
output_path = Path("people_detected_image.jpg")
cv2.imwrite(output_path, frame)
print(f"Detection done. Image saved as {output_path}")
cv2.waitKey(0)
cv2.destroyAllWindows()
