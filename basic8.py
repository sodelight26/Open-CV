import cv2
import torch
from pathlib import Path

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo='check')

# Set the video file path
video_path = 'video2.mp4'

# Initialize the video capture
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects using YOLOv5
    results = model(frame)

    # Access detected objects and their properties
    labels = results.names  # Get class labels
    pred = results.pred[0]  # Get the first image's predictions

    for det in pred:
        x1, y1, x2, y2, conf, class_idx = det.tolist()
        class_name = labels[int(class_idx)]

        if conf > 0.5:  # Adjust confidence threshold as needed
            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'{class_name}: {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with detections
    cv2.imshow('YOLOv5 Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
