from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path, conf=0.25, imgsz=640):
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz

    def detect(self, frame):
        results = self.model(
            frame,
            conf=self.conf,
            imgsz=self.imgsz,
            classes=[0],     # person only
            verbose=False    # clean terminal
        )
        return results