import cv2
import torch
from ultralytics import YOLO

# Load YOLO model
model = YOLO("datasets/europe/weights/best.pt")

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()
print(f"Using device: {device}")

# Open video file or webcam (change 'video.mp4' to 0 for webcam)
video_path = "Frankfurt_TS_Video.mp4"  # Change to 0 for webcam
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define output video writer
output_path = "analyzed_output.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Use "XVID" for .avi format
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Set default confidence and IOU thresholds
conf = 0.75  # Confidence threshold
iou = 0.9   # IOU threshold

def process_frame(frame):
    """Processes a single frame."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run detection on the frame
    results = model(frame_rgb, conf=conf, iou=iou, verbose=True, imgsz=1920, batch=2, workers=4) # batch -> limitations for GPU, workers -> limitations for CPU

    # Draw results on the frame
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf_score = box.conf[0].item()
            class_id = int(box.cls[0])
            label = model.names[class_id]

            # Draw rectangle and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({conf_score:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame

# Process frames and save the output
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process the frame
    processed_frame = process_frame(frame)

    # Write the processed frame to output video
    out.write(processed_frame)
    #cv2.imshow("Detections", processed_frame)
    #cv2.waitKey(1)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
