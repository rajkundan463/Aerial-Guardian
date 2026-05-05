import cv2
import time
from detector.yolo_detector import YOLODetector
from utils.video_utils import get_video_writer
import config


def main():
    # Use webcam (for testing) OR replace with video file later
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Camera not accessible")
        return

    # Force resolution (camera)
    cap.set(3, 1280)
    cap.set(4, 720)

    # Output resolution (must match resized frame)
    output_width = 960
    output_height = 540

    out = get_video_writer(config.VIDEO_OUTPUT, output_width, output_height)

    detector = YOLODetector(
        config.MODEL_PATH,
        config.CONF_THRESHOLD,
        config.IMG_SIZE
    )

    warmup_frames = 10
    frame_count = 0
    timed_frames = 0
    start_time = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for speed-performance balance
        frame = cv2.resize(frame, (output_width, output_height))

        # Detection
        results = detector.detect(frame)

        annotated = frame.copy()

        # Safe detection + manual bounding boxes
        if results[0].boxes is not None:
            for box in results[0].boxes:
                cls = int(box.cls[0])
                if cls != 0:  # Only person class
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(annotated, f"Person {conf:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 0, 0), 2)

        # Warmup phase (skip FPS calc)
        if frame_count < warmup_frames:
            frame_count += 1
        else:
            if start_time is None:
                start_time = time.time()

            timed_frames += 1

            # Live FPS calculation
            current_time = time.time()
            fps = timed_frames / (current_time - start_time)

            #Draw FPS BEFORE writing (so it appears in saved video)
            cv2.putText(annotated, f"FPS: {fps:.2f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2)

        # Write output video
        out.write(annotated)

        # Display
        cv2.imshow("Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Final FPS print
    if start_time is not None:
        total_time = time.time() - start_time
        final_fps = timed_frames / total_time if total_time > 0 else 0
        print(f"Average FPS (after warmup): {final_fps:.2f}")
    else:
        print("Not enough frames for FPS calculation")

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()