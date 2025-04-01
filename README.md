# Traffic_Sign_Detection_Germany
Traffic Sign Detection, country Germany


dataset link: https://www.kaggle.com/datasets/pkdarabi/cardetection/data

# **Пример config.yaml**
```yaml
model:
  path: "416_100epoch_yolov8s_model/weights/best.pt"
  conf: 0.5
  iou: 0.45
  # Необязательные параметры (можно закомментировать)
  imgsz: 1280
  augment: False
  device: "auto"  # или "cuda"/"cpu"
video_params:
  video_input: "docker/Frankfurt_TS_Video.mp4"
  video_output: "output.mp4"
```