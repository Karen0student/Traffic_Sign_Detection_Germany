import logging
from logging.handlers import QueueHandler, QueueListener
import sys
import multiprocessing as mp
from multiprocessing import Process, Queue, Event
import time
import queue
import os
from collections import OrderedDict
import yaml
import cv2
import torch
from ultralytics import YOLO


# Настройка системы логирования
def setup_logging():
    """Настройка системы логирования для главного процесса и дочерних"""
    # Создаем очередь для логов
    log_queue = mp.Queue()

    # Форматтер для всех обработчиков
    formatter = logging.Formatter(
        "%(asctime)s - %(processName)s - %(levelname)s - %(message)s"
    )

    # Обработчики (консоль и файл)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler("app.log")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    # Слушатель очереди (работает в главном процессе)
    listener = QueueListener(
        log_queue, console_handler, file_handler, respect_handler_level=True
    )
    listener.start()

    # Настройка корневого логгера для главного процесса
    main_logger = logging.getLogger()
    main_logger.setLevel(logging.DEBUG)
    main_logger.addHandler(console_handler)
    main_logger.addHandler(file_handler)

    return log_queue, listener


# Инициализация логгера для процессов
def setup_worker_logger(log_queue):
    """Настройка логгера для worker-процессов"""
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Очистка существующих обработчиков
    for h in logger.handlers[:]:
        logger.removeHandler(h)

    # Добавляем обработчик очереди
    queue_handler = QueueHandler(log_queue)
    logger.addHandler(queue_handler)

    return logger


def load_config(config_path="config.yaml"):
    """Загрузка конфига с обработкой необязательных полей"""
    logger = logging.getLogger()
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        model_cfg = config.get("model", {})
        video_cfg = config.get("video_params", {})
        app_params = config.get("app_params", {})

        device = model_cfg.get("device", "auto")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Auto-selected device: {device}")

        model_params = {
            "model_path": model_cfg["path"],
            "conf": model_cfg.get("conf", 0.25),
            "iou": model_cfg.get("iou", 0.45),
            "device": device,
            "imgsz": model_cfg.get("imgsz"),
            "augment": model_cfg.get("augment"),
        }
        model_params = {k: v for k, v in model_params.items() if v is not None}

        logger.debug("Config loaded successfully")
        return {
            "model_params": model_params,
            "video_params": video_cfg,
            "app_params": app_params,
        }

    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        raise


def writer_process(
    output_queue, output_path, frame_size, fps, total_frames, stop_event, log_queue
):
    """Процесс записи видео с логированием"""
    setup_worker_logger(log_queue)
    logger = logging.getLogger()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    expected_idx = 0
    frame_buffer = OrderedDict()
    last_log_time = time.time()

    try:
        logger.info(f"Writer started for {output_path}")
        while not stop_event.is_set() or expected_idx < total_frames:
            try:
                data = output_queue.get(timeout=1)
                if data == "DONE":
                    logger.info("Received DONE signal")
                    break

                frames_to_process = [data] if isinstance(data, tuple) else data

                for frame_idx, frame in frames_to_process:
                    frame_buffer[frame_idx] = frame

                while expected_idx in frame_buffer:
                    out.write(frame_buffer.pop(expected_idx))
                    expected_idx += 1

                    if time.time() - last_log_time > 1.0:
                        logger.info(
                            f"Written {expected_idx}/{total_frames} frames ({expected_idx/total_frames:.1%})"
                        )
                        last_log_time = time.time()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Writer error: {str(e)}")
                break

    finally:
        while expected_idx < total_frames and expected_idx in frame_buffer:
            out.write(frame_buffer.pop(expected_idx))
            expected_idx += 1

        out.release()
        logger.info(f"Writer finished. Total written: {expected_idx}")


def process_batch(batch, model, model_params):
    """Обработка батча с логированием"""
    logger = logging.getLogger()
    try:
        indices, frames = zip(*sorted(batch, key=lambda x: x[0]))

        with torch.no_grad():
            results = model(
                frames,
                imgsz=model_params.get("imgsz"),
                conf=model_params["conf"],
                iou=model_params["iou"],
                augment=model_params.get("augment", False),
                verbose=False,
            )

        processed = []
        for idx, frame, result in zip(indices, frames, results):
            output_frame = frame.copy()

            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)

                for i, (box, conf, cls_id) in enumerate(zip(boxes, confs, class_ids)):
                    x1, y1, x2, y2 = map(int, box)
                    color = (0, 255, 0)
                    cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, thickness=2)
                    label = f"{model.names[cls_id]} {conf:.2f}"
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

        logger.debug(f"Processed batch with {len(batch)} frames")
        return processed

    except Exception as e:
        logger.error(f"Error processing batch: {str(e)}")
        return [(idx, frame.copy()) for idx, frame in batch]


def worker(
    frame_queue,
    output_queue,
    worker_id,
    model_params,
    stop_event,
    log_queue,
    max_retries,
    batch_size,
    max_output_queue_size,
):
    """Worker-процесс с логированием"""
    setup_worker_logger(log_queue)
    logger = logging.getLogger()
    logger.info(f"Worker {worker_id} started")

    model = YOLO(model_params["model_path"])
    model.to(model_params["device"]).eval()
    batch = []

    while not stop_event.is_set():
        try:
            while len(batch) < batch_size:
                frame_data = frame_queue.get(timeout=1)
                if frame_data is None:
                    if batch:
                        try:
                            processed = process_batch(batch, model, model_params)
                            output_queue.put(processed)
                        except Exception as e:
                            logger.warning(f"Final batch failed: {str(e)}")
                            output_queue.put(
                                [(idx, frame.copy()) for idx, frame in batch]
                            )
                    logger.info("Worker received exit signal")
                    return
                batch.append(frame_data)

            for attempt in range(max_retries):
                try:
                    processed = process_batch(batch, model, model_params)
                    while output_queue.qsize() > max_output_queue_size:
                        time.sleep(0.1)
                    output_queue.put(processed)
                    batch = []
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(
                            f"Batch failed after {max_retries} attempts: {str(e)}"
                        )
                        output_queue.put([(idx, frame.copy()) for idx, frame in batch])
                        batch = []
                    time.sleep(0.1)

        except queue.Empty:
            if batch:
                try:
                    processed = process_batch(batch, model, model_params)
                    output_queue.put(processed)
                    batch = []
                except Exception as e:
                    logger.error(f"Partial batch failed: {str(e)}")
                    output_queue.put([(idx, frame.copy()) for idx, frame in batch])
                    batch = []
            continue

    output_queue.put((None, worker_id))


def main():
    # Инициализация системы логирования
    log_queue, listener = setup_logging()
    logger = logging.getLogger(__name__)  # Используем именованный логгер

    try:
        logger.info("=" * 50)
        logger.info("Application started")
        logger.info("=" * 50)

        if len(sys.argv) < 2:
            logger.error("Usage: python script.py <config_path>")
            return

        config_path = sys.argv[1]
        logger.info(f"Loading config from {config_path}")

        _params = load_config(config_path)
        model_params = _params["model_params"]
        video_params = _params["video_params"]
        app_params = _params["app_params"]

        logger.info(f"Model params: {model_params}")
        logger.info(f"Video params: {video_params}")
        logger.info(f"App params: {app_params}")

        cap = cv2.VideoCapture(video_params["video_input"])
        if not cap.isOpened():
            logger.error("Could not open video file")
            return

        frame_size = (int(cap.get(3)), int(cap.get(4)))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(
            f"Video info: {frame_size} @ {fps}fps, total frames: {total_frames}"
        )

        frame_queue = Queue(maxsize=app_params["frame_queue_size"])
        output_queue = Queue()
        stop_event = Event()

        workers = []
        for i in range(app_params["num_workers"]):
            p = Process(
                target=worker,
                args=(
                    frame_queue,
                    output_queue,
                    i,
                    model_params,
                    stop_event,
                    log_queue,
                    app_params["max_retries"],
                    app_params["batch_size"],
                    app_params["max_output_queue_size"],
                ),
                name=f"Worker-{i}",
            )
            p.start()
            workers.append(p)
            logger.debug(f"Started worker process {i}")

        writer = Process(
            target=writer_process,
            args=(
                output_queue,
                video_params["video_output"],
                frame_size,
                fps,
                total_frames,
                stop_event,
                log_queue,
            ),
            name="Writer",
        )
        writer.start()
        logger.debug("Started writer process")

        frame_idx = 0
        start_time = time.time()
        last_log_time = start_time

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.info("Reached end of video")
                    break

                while frame_queue.qsize() > app_params["frame_queue_size"] * 0.8:
                    time.sleep(0.1)

                frame_queue.put((frame_idx, frame))
                frame_idx += 1

                if time.time() - last_log_time > 1.0:
                    elapsed = time.time() - start_time
                    current_fps = frame_idx / elapsed if elapsed > 0 else 0
                    logger.info(
                        f"Sent {frame_idx}/{total_frames} frames ({current_fps:.2f} FPS)"
                    )
                    last_log_time = time.time()

            logger.info("All frames sent, waiting for workers...")
            for _ in range(app_params["num_workers"]):
                frame_queue.put(None)

            active_workers = app_params["num_workers"]
            while active_workers > 0:
                try:
                    msg = output_queue.get(timeout=5)
                    if isinstance(msg, tuple) and msg[0] is None:
                        active_workers -= 1
                        logger.info(f"Worker {msg[1]} finished")
                except queue.Empty:
                    logger.warning("Timeout waiting for workers")
                    break

            output_queue.put("DONE")
            writer.join(timeout=10)
            logger.info("Writer process finished")

        except KeyboardInterrupt:
            logger.warning("Keyboard interrupt received")
        finally:
            stop_event.set()
            cap.release()

            for p in workers:
                if p.is_alive():
                    p.terminate()
            if writer.is_alive():
                writer.terminate()

            elapsed = time.time() - start_time
            logger.info(
                f"Processing stats:\n"
                f"- Total frames: {total_frames}\n"
                f"- Processed: {frame_idx}\n"
                f"- Avg FPS: {frame_idx/elapsed:.2f}\n"
                f"- Time: {elapsed:.2f}s"
            )

    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}", exc_info=True)
    finally:
        listener.stop()
        logging.shutdown()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    torch.backends.cudnn.benchmark = True
    main()
