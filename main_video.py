import cv2
import torch
import numpy as np
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor

# Load YOLO model
model = YOLO("datasets/europe/runs/detect/train3/weights/best.pt", verbose=False)

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
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
output_path = "analyzed_output_3.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Use "XVID" for .avi format
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Set filters
tile_size = 256
conf = 0.6
iou = 0.6 # is a metric used to evaluate the accuracy of object detection

def tile_frame(frame, tile_size, overlap=0.2): # overlap ensures objects split between tiles remain visible in multiple tiles 
    """Splits frame into overlapping tiles."""
    height, width, _ = frame.shape
    step = int(tile_size * (1 - overlap))
    tiles, positions = [], []
    
    for y in range(0, height - tile_size + 1, step):
        for x in range(0, width - tile_size + 1, step):
            tile = frame[y:y + tile_size, x:x + tile_size]
            tiles.append(tile)
            positions.append((x, y))
    
    return tiles, positions

def detect_on_tiles(model, tiles, tile_size, conf, iou):
    """Runs batch detection on multiple tiles at once for GPU efficiency."""
    results = model(tiles, conf=conf, imgsz=tile_size, iou=iou, verbose=False)  # Batch inference
    detections = []
    
    for i, result in enumerate(results):
        x_offset, y_offset = tile_positions[i]  # Match tile position
        
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf_score = box.conf[0].item()
            class_id = int(box.cls[0])
            label = model.names[class_id]
            
            x1 += x_offset
            x2 += x_offset
            y1 += y_offset
            y2 += y_offset
            
            detections.append((x1, y1, x2, y2, conf_score, label))
    
    return detections

def process_frame(frame, model, tile_size, conf, iou):
    """Processes a single frame in a separate thread."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tiles, positions = tile_frame(frame_rgb, tile_size)
    
    global tile_positions  # Store positions globally for batch processing
    tile_positions = positions
    
    # Run batch detection on tiles
    detections = detect_on_tiles(model, tiles, tile_size, conf, iou)
    
    for x1, y1, x2, y2, conf, label in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame

executor = ThreadPoolExecutor(max_workers=10)
future_frames = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Submit frame processing to thread pool
    future = executor.submit(process_frame, frame, model, tile_size, conf, iou)
    future_frames.append(future)

    # Display and save processed frames as they finish
    if len(future_frames) > 5:  # Avoid excessive memory usage
        future = future_frames.pop(0)
        processed_frame = future.result()
        #cv2.imshow("YOLO Video Detection (Multithreaded)", processed_frame)
        out.write(processed_frame)  # Save frame to video
    
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# Process remaining frames
# for future in future_frames:
#     processed_frame = future.result()
#     cv2.imshow("YOLO Video Detection (Multithreaded)", processed_frame)
#     out.write(processed_frame)  # Save remaining frames

cap.release()
out.release()
cv2.destroyAllWindows()
executor.shutdown()
