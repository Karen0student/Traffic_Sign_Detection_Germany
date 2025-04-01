# Traffic_Sign_Detection_Germany
Traffic Sign Detection, country Germany


dataset link: https://www.kaggle.com/datasets/pkdarabi/cardetection/data

# **Пример config.yaml**
```yaml
model:
  path: "path/to/model.pt"
  conf: 0.5
  iou: 0.45
  # Необязательные параметры (можно закомментировать)
  imgsz: 1280
  augment: False
  device: "auto"  # или "cuda"/"cpu"
video_params:
  video_input: "video.mp4"
  video_output: "output.mp4"
app_params:
  num_workers: 8
  frame_queue_size: 15
  batch_size: 4
  max_retries: 3
  max_output_queue_size: 20
```