import cv2
import time
import os
from detector.yolo_detector import YOLODetector
from utils.video_utils import get_video_writer
import config
from tracker.simple_tracker import SimpleTracker


def main():

    sequence_path = config.SEQUENCE_PATH
    tracker = SimpleTracker(max_distance=50)


    image_files = sorted(os.listdir(sequence_path))

    # Output video writer
    out = get_video_writer(
        config.VIDEO_OUTPUT,
        config.OUTPUT_WIDTH,
        config.OUTPUT_HEIGHT
    )

    # Detector
    detector = YOLODetector(
        config.MODEL_PATH,
        config.CONF_THRESHOLD,
        config.IMG_SIZE
    )


    # Performance Parameters

    warmup_frames = 10
    frame_count = 0
    timed_frames = 0
    start_time = None

    # Skip frames for better FPS
    SKIP_FRAMES = 2


    # Main Processing Loop

    for img_name in image_files:

        frame_count += 1

        #  Frame skipping optimization
        if frame_count % SKIP_FRAMES != 0:
            continue

        img_path = os.path.join(sequence_path, img_name)

        frame = cv2.imread(img_path)

        if frame is None:
            continue

        # Resize frame
        frame = cv2.resize(
            frame,
            (config.OUTPUT_WIDTH, config.OUTPUT_HEIGHT)
        )

  
        # Detection

        results = detector.detect(frame)

        annotated = frame.copy()

        # Store detections for tracker
        detections = []


        # Draw Detection Boxes

        if results[0].boxes is not None:

            for box in results[0].boxes:

                cls = int(box.cls[0])

                # Person class only
                if cls != 0:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                conf = float(box.conf[0])

                # Add to tracker detections
                detections.append([x1, y1, x2, y2])

                # Confidence-based color
                color = (
                    (0, 255, 0)
                    if conf > 0.5
                    else (0, 165, 255)
                )

                # Detection box
                cv2.rectangle(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                # Confidence label
                cv2.putText(
                    annotated,
                    f"Person {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )

         
        # Tracking Update
         
        tracks = tracker.update(detections)

         
        # Draw Tracking IDs + Trajectory
         
        for track in tracks:

            x1, y1, x2, y2 = map(int, track.bbox)

            track_id = track.id

            # Tracking box
            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

            # ID label
            cv2.putText(
                annotated,
                f"ID {track_id}",
                (x1, y1 - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )

            # Trajectory lines
            for i in range(1, len(track.history)):

                pt1 = tuple(track.history[i - 1].astype(int))
                pt2 = tuple(track.history[i].astype(int))

                cv2.line(
                    annotated,
                    pt1,
                    pt2,
                    (0, 255, 255),
                    2
                )

         
        # FPS Calculation
         
        if frame_count < warmup_frames:

            continue

        if start_time is None:
            start_time = time.time()

        timed_frames += 1

        elapsed_time = time.time() - start_time

        fps = (
            timed_frames / elapsed_time
            if elapsed_time > 0
            else 0
        )

        # FPS Overlay
        cv2.putText(
            annotated,
            f"FPS: {fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

         
        # Save + Display
         
        out.write(annotated)

        cv2.imshow("Aerial Guardian - VisDrone MOT", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

     
    # Final FPS Summary
     
    if start_time is not None:

        total_time = time.time() - start_time

        final_fps = (
            timed_frames / total_time
            if total_time > 0
            else 0
        )

        print(f"\n Average FPS: {final_fps:.2f}")

    # Cleanup
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()