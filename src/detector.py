import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Dict, Any, Tuple

class ONNXDetector:
    def __init__(
        self,
        model_path: str = "models/best.onnx",
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        input_size: Tuple[int, int] = (640, 640)
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size

        # Exact class map matching your dataset
        self.class_names = {
            0: "boots",
            1: "gloves",
            2: "goggles",
            3: "helmet",
            4: "person",
            5: "vest"
        }

        # Multi-threaded CPU execution configuration
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """Letterbox resize image to 640x640 with aspect-ratio padding."""
        h, w = image.shape[:2]
        r = min(self.input_size[0] / h, self.input_size[1] / w)
        unpad_w, unpad_h = int(round(w * r)), int(round(h * r))
        dw, dh = (self.input_size[1] - unpad_w) / 2, (self.input_size[0] - unpad_h) / 2

        resized = cv2.resize(image, (unpad_w, unpad_h), interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        blob = padded[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        blob = np.ascontiguousarray(blob, dtype=np.float32) / 255.0
        return np.expand_dims(blob, axis=0), r, (dw, dh)

    def _postprocess(
        self, output: np.ndarray, ratio: float, pad: Tuple[float, float], orig_shape: Tuple[int, int]
    ) -> List[Dict[str, Any]]:
        """Parses output [1, 10, 8400] -> [8400, 10] into filtered bounding boxes."""
        preds = np.squeeze(output[0]).T  # shape: (8400, 4 + 6_classes)

        boxes = preds[:, :4]
        scores = preds[:, 4:]

        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)

        mask = confidences >= self.conf_threshold
        boxes, confidences, class_ids = boxes[mask], confidences[mask], class_ids[mask]

        if len(boxes) == 0:
            return []

        dw, dh = pad
        x1 = (boxes[:, 0] - boxes[:, 2] / 2 - dw) / ratio
        y1 = (boxes[:, 1] - boxes[:, 3] / 2 - dh) / ratio
        x2 = (boxes[:, 0] + boxes[:, 2] / 2 - dw) / ratio
        y2 = (boxes[:, 1] + boxes[:, 3] / 2 - dh) / ratio

        orig_h, orig_w = orig_shape
        x1, y1 = np.clip(x1, 0, orig_w), np.clip(y1, 0, orig_h)
        x2, y2 = np.clip(x2, 0, orig_w), np.clip(y2, 0, orig_h)

        xywh_boxes = [[int(x), int(y), int(x2[i] - x), int(y2[i] - y)] for i, (x, y) in enumerate(zip(x1, y1))]
        indices = cv2.dnn.NMSBoxes(xywh_boxes, confidences.tolist(), self.conf_threshold, self.iou_threshold)

        detections = []
        if len(indices) > 0:
            for idx in indices.flatten():
                detections.append({
                    "bbox": [float(x1[idx]), float(y1[idx]), float(x2[idx]), float(y2[idx])],
                    "confidence": float(confidences[idx]),
                    "class_id": int(class_ids[idx]),
                    "class_name": self.class_names.get(int(class_ids[idx]), "unknown")
                })
        return detections

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        blob, ratio, pad = self._preprocess(frame)
        outputs = self.session.run([self.output_name], {self.input_name: blob})
        return self._postprocess(outputs, ratio, pad, frame.shape[:2])