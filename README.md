# Traffic_Sign_Detection_Germany
Traffic Sign Detection, country Germany


dataset link: https://www.kaggle.com/datasets/pkdarabi/cardetection/data

# **example config.yaml**
```yaml
model:
  path: "path/to/model.pt"
  conf: 0.5
  iou: 0.45
  imgsz: 1280
  augment: False
  device: "auto"  # or "cuda"/"cpu"
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


to run in docker, simply execute bash docker/docker_container_run.sh, but before that, you should add video path to connect that volume into container.

!!! config.yaml is different for docker