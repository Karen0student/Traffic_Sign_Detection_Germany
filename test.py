import cv2
import torch
from ultralytics import YOLO
import sys
import multiprocessing as mp
from multiprocessing import Process, Queue
import time

# Настройки мультипроцессинга
NUM_PROCESSES = 4  # Количество процессов (можно настроить под ваше железо)
FRAME_QUEUE_SIZE = 10  # Максимальный размер очереди кадров


# Загрузка модели YOLO (будет загружена в каждом процессе)
def load_model():
    model = YOLO("416_100epoch_yolov8s_model/weights/best.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    print(f"Using device: {device}")
    return model


# Функция обработки кадра (работает в отдельных процессах)
def process_frames(frame_queue, output_queue, process_id):
    print(f"Process {process_id} started")
    model = load_model()

    conf = 0.75  # Confidence threshold
    iou = 0.9  # IOU threshold

    while True:
        frame_data = frame_queue.get()
        if frame_data is None:  # Сигнал завершения
            break

        frame_idx, frame = frame_data
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Запуск детекции
        results = model(frame_rgb, conf=conf, iou=iou, verbose=False, imgsz=1920)

        # Отрисовка результатов
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf_score = box.conf[0].item()
                class_id = int(box.cls[0])
                label = model.names[class_id]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{label} ({conf_score:.2f})",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

        output_queue.put((frame_idx, frame))

    print(f"Process {process_id} finished")


# Главная функция
def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <video_path>")
        return

    video_path = sys.argv[1]
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Получение параметров видео
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Создание выходного видео
    output_path = "test_analyzed_output_1920.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Очереди для обмена данными между процессами
    frame_queue = Queue(maxsize=FRAME_QUEUE_SIZE)
    output_queue = Queue()

    # Запуск процессов обработки
    processes = []
    for i in range(NUM_PROCESSES):
        p = Process(target=process_frames, args=(frame_queue, output_queue, i))
        p.start()
        processes.append(p)

    # Чтение и распределение кадров
    frame_idx = 0
    processed_frames = 0
    frame_buffer = {}

    start_time = time.time()

    try:
        while True:
            # Чтение кадра
            ret, frame = cap.read()
            if not ret:
                break

            # Добавление кадра в очередь (блокируется если очередь полна)
            frame_queue.put((frame_idx, frame))
            frame_idx += 1

            # Проверка обработанных кадров
            while not output_queue.empty():
                idx, processed_frame = output_queue.get()
                frame_buffer[idx] = processed_frame

                # Запись кадров в порядке их следования
                while processed_frames in frame_buffer:
                    out.write(frame_buffer[processed_frames])
                    del frame_buffer[processed_frames]
                    processed_frames += 1

                    # Вывод прогресса
                    if processed_frames % 10 == 0:
                        elapsed = time.time() - start_time
                        fps = processed_frames / elapsed
                        print(
                            f"Processed {processed_frames}/{total_frames} frames, {fps:.2f} FPS"
                        )

    except KeyboardInterrupt:
        print("Interrupted by user")

    # Завершение процессов
    for _ in range(NUM_PROCESSES):
        frame_queue.put(None)

    for p in processes:
        p.join()

    # Запись оставшихся кадров
    while processed_frames < frame_idx:
        if processed_frames in frame_buffer:
            out.write(frame_buffer[processed_frames])
            del frame_buffer[processed_frames]
            processed_frames += 1

    # Освобождение ресурсов
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("Processing completed")


if __name__ == "__main__":
    mp.set_start_method("spawn")  # Важно для CUDA и PyTorch
    main()
