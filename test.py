import cv2
import torch
from ultralytics import YOLO
import sys
import multiprocessing as mp
from multiprocessing import Process, Queue, Event
import time
import queue
import os
from collections import OrderedDict

# Настройки
NUM_WORKERS = 8
FRAME_QUEUE_SIZE = 15
BATCH_SIZE = 4
MAX_RETRIES = 3
MAX_OUTPUT_QUEUE_SIZE = 20  # Ограничение для избежания переполнения памяти


def writer_process(
    output_queue, output_path, frame_size, fps, total_frames, stop_event
):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    # Для сохранения порядка кадров
    expected_idx = 0
    frame_buffer = OrderedDict()
    last_log_time = time.time()

    try:
        while not stop_event.is_set() or expected_idx < total_frames:
            try:
                # Получаем данные из очереди
                data = output_queue.get(timeout=1)
                if data == "DONE":
                    break

                # Обрабатываем как отдельные кадры, так и батчи
                frames_to_process = []
                if isinstance(data, tuple):  # Одиночный кадр
                    frames_to_process = [data]
                elif isinstance(data, list):  # Батч кадров
                    frames_to_process = data

                # Добавляем кадры в буфер
                for frame_idx, frame in frames_to_process:
                    frame_buffer[frame_idx] = frame

                # Записываем кадры по порядку
                while expected_idx in frame_buffer:
                    out.write(frame_buffer.pop(expected_idx))
                    expected_idx += 1

                    # Логирование прогресса
                    if time.time() - last_log_time > 1.0:
                        print(
                            f"Written {expected_idx}/{total_frames} frames ({expected_idx/total_frames:.1%})"
                        )
                        last_log_time = time.time()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Writer error: {str(e)}")
                break

    finally:
        # Записываем оставшиеся кадры
        while expected_idx < total_frames and expected_idx in frame_buffer:
            out.write(frame_buffer.pop(expected_idx))
            expected_idx += 1

        out.release()
        print(f"Writer finished. Total written: {expected_idx}")


def preprocess_frame(frame, method="none"):
    """Баланс качества и скорости"""
    if method == "none":
        return frame

    # Быстрая нормализация (5-7% slowdown)
    if method == "fast_norm":
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.normalize(l, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    # Только для критичных случаев (15-20% slowdown)
    if method == "full_enhance":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def process_batch(batch, model, model_params):
    """Обработка батча с раздельной отрисовкой каждого bbox"""
    # 1. Сортировка кадров по индексам
    indices, frames = zip(*sorted(batch, key=lambda x: x[0]))

    # 2. Векторизованная детекция объектов
    with torch.no_grad():
        results = model(
            frames,
            imgsz=model_params["imgsz"],
            conf=model_params["conf"],
            iou=model_params["iou"],
            augment=False,
            verbose=False,
        )

    processed = []
    for idx, frame, result in zip(indices, frames, results):
        # Создаем копию кадра для отрисовки
        output_frame = frame.copy()

        if result.boxes is not None:
            # Извлекаем данные детекции
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)

            # 3. Индивидуальная отрисовка КАЖДОГО bbox
            for i, (box, conf, cls_id) in enumerate(zip(boxes, confs, class_ids)):
                x1, y1, x2, y2 = map(int, box)

                # Генерация уникального цвета для каждого объекта
                color = (0, 255, 0)

                # Отрисовка прямоугольника
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, thickness=2)

                # Подготовка текстовой метки
                label = f"{model.names[cls_id]} {conf:.2f}"

                # Текст с контрастным цветом
                cv2.putText(
                    output_frame,
                    label,
                    (x1, y1 - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        processed.append((idx, output_frame))

    return processed


def worker(frame_queue, output_queue, worker_id, model_params, stop_event):
    model = YOLO(model_params["model_path"])
    model.to(model_params["device"]).eval()
    batch = []

    while not stop_event.is_set():
        try:
            # Собираем батч кадров
            while len(batch) < BATCH_SIZE:
                frame_data = frame_queue.get(timeout=1)
                if frame_data is None:  # Сигнал завершения
                    if batch:
                        try:
                            processed = process_batch(batch, model, model_params)
                            output_queue.put(processed)
                        except:
                            output_queue.put(
                                [(idx, frame.copy()) for idx, frame in batch]
                            )
                    return
                batch.append(frame_data)

            # Обработка с повторами при ошибках
            for attempt in range(MAX_RETRIES):
                try:
                    processed = process_batch(batch, model, model_params)

                    # Контроль размера очереди вывода
                    while output_queue.qsize() > MAX_OUTPUT_QUEUE_SIZE:
                        time.sleep(0.1)

                    output_queue.put(processed)
                    batch = []
                    break
                except Exception as e:
                    if attempt == MAX_RETRIES - 1:
                        print(
                            f"Worker {worker_id} batch failed after {MAX_RETRIES} attempts"
                        )
                        output_queue.put([(idx, frame.copy()) for idx, frame in batch])
                        batch = []
                    time.sleep(0.1)

        except queue.Empty:
            if batch:  # Отправляем неполный батч
                try:
                    processed = process_batch(batch, model, model_params)
                    output_queue.put(processed)
                    batch = []
                except:
                    output_queue.put([(idx, frame.copy()) for idx, frame in batch])
                    batch = []
            continue

    output_queue.put((None, worker_id))


def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <video_path>")
        return

    model_params = {
        "model_path": "416_100epoch_yolov8s_model/weights/best.pt",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "conf": 0.5,
        "iou": 0.45,
        "imgsz": 1280,
        "augment": False,
    }

    video_path = sys.argv[1]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_size = (int(cap.get(3)), int(cap.get(4)))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Очереди и управление
    frame_queue = Queue(maxsize=FRAME_QUEUE_SIZE)
    output_queue = Queue()
    stop_event = Event()

    # Запуск процессов
    workers = []
    for i in range(NUM_WORKERS):
        p = Process(
            target=worker, args=(frame_queue, output_queue, i, model_params, stop_event)
        )
        p.start()
        workers.append(p)

    writer = Process(
        target=writer_process,
        args=(
            output_queue,
            "output.mp4",
            frame_size,
            fps,
            total_frames,
            stop_event,
        ),
    )
    writer.start()

    # Чтение и отправка кадров с сохранением порядка
    frame_idx = 0
    start_time = time.time()
    last_log_time = start_time

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Контроль размера очереди ввода
            while frame_queue.qsize() > FRAME_QUEUE_SIZE * 0.8:
                time.sleep(0.1)

            frame_queue.put((frame_idx, frame))
            frame_idx += 1

            # Логирование прогресса
            if time.time() - last_log_time > 1.0:
                elapsed = time.time() - start_time
                current_fps = frame_idx / elapsed if elapsed > 0 else 0
                print(f"Sent {frame_idx}/{total_frames} frames ({current_fps:.2f} FPS)")
                last_log_time = time.time()

        # Завершение работы
        print("\nAll frames sent. Waiting for completion...")
        for _ in range(NUM_WORKERS):
            frame_queue.put(None)

        # Ожидание завершения
        active_workers = NUM_WORKERS
        while active_workers > 0:
            try:
                msg = output_queue.get(timeout=5)
                if isinstance(msg, tuple) and msg[0] is None:
                    active_workers -= 1
                    print(f"Worker {msg[1]} finished")
            except queue.Empty:
                print("Timeout waiting for workers")
                break

        output_queue.put("DONE")
        writer.join(timeout=10)

    except KeyboardInterrupt:
        print("\nStopping early...")
    finally:
        stop_event.set()
        cap.release()

        # Принудительное завершение
        for p in workers:
            p.terminate()
        writer.terminate()

        # Статистика
        elapsed = time.time() - start_time
        print(f"\nProcessing stats:")
        print(f"- Total frames: {total_frames}")
        print(f"- Frames processed: {frame_idx}")
        print(f"- Average FPS: {frame_idx/elapsed:.2f}")
        print(f"- Total time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    torch.backends.cudnn.benchmark = True
    main()
