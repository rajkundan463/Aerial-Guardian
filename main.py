import cv2
import time
import os
from detector.yolo_detector import YOLODetector
from utils.video_utils import get_video_writer
import config


def main():
    sequence_path = config.SEQUENCE_PATH

    # Load all frames
    image_files = sorted(os.listdir(sequence_path))

    out = get_video_writer(
        config.VIDEO_OUTPUT,
        config.OUTPUT_WIDTH,
        config.OUTPUT_HEIGHT
    )

    detector = YOLODetector(
        config.MODEL_PATH,
        config.CONF_THRESHOLD,
        config.IMG_SIZE
    )

    warmup_frames = 10
    frame_count = 0
    timed_frames = 0
    start_time = None

    for img_name in image_files:
        img_path = os.path.join(sequence_path, img_name)

        frame = cv2.imread(img_path)
        if frame is None:
            continue

        # Resize for processing
        frame = cv2.resize(frame, (config.OUTPUT_WIDTH, config.OUTPUT_HEIGHT))

        # Detection
        results = detector.detect(frame)

        annotated = frame.copy()

        # Draw detections
        if results[0].boxes is not None:
            for box in results[0].boxes:
                cls = int(box.cls[0])
                if cls != 0:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                color = (0, 255, 0) if conf > 0.5 else (0, 165, 255)

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, f"{conf:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 2)

        # FPS calculation
        if frame_count < warmup_frames:
            frame_count += 1
        else:
            if start_time is None:
                start_time = time.time()

            timed_frames += 1
            fps = timed_frames / (time.time() - start_time)

            cv2.putText(annotated, f"FPS: {fps:.2f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2)

        # Save video
        out.write(annotated)

        cv2.imshow("VisDrone Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if start_time is not None:
        total_time = time.time() - start_time
        final_fps = timed_frames / total_time
        print(f"Average FPS: {final_fps:.2f}")

    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()