from ultralytics import YOLO
import cv2

class YOLODetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, frame, visualize=False):
        results = self.model(frame, verbose=False)[0]
        height, width, _ = frame.shape
        detections = []

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0])

            if cls == 0 and conf > 0.6:  # Only person, confidence threshold
                # Shrink the box slightly to cut background
                pad = 0.05
                w = x2 - x1
                h = y2 - y1
                x1_new = max(int(x1 + pad * w), 0)
                y1_new = max(int(y1 + pad * h), 0)
                x2_new = min(int(x2 - pad * w), width - 1)
                y2_new = min(int(y2 - pad * h), height - 1)

                detections.append([x1_new, y1_new, x2_new, y2_new, conf])

                if visualize:
                    cv2.rectangle(frame, (x1_new, y1_new), (x2_new, y2_new), (0, 0, 255), 2)
                    cv2.putText(frame, f"{conf:.2f}", (x1_new, y1_new - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        return detections
