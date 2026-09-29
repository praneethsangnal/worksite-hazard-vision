import os
import time
import cv2
import numpy as np
from src.detector import ONNXDetector

def test_module_1():
    model_path = "models/best.onnx"
    assert os.path.exists(model_path), f"Error: {model_path} not found. Ensure you downloaded best.onnx into models/"

    print("[1/4] Initializing ONNXDetector on CPU...")
    detector = ONNXDetector(model_path=model_path, conf_threshold=0.30)
    print("      Model loaded successfully into ONNX Runtime!")

    # Check for test image or generate a dummy frame
    test_image_path = "test.jpg"
    if not os.path.exists(test_image_path):
        print("[2/4] 'test.jpg' not found. Creating a synthetic frame for pipeline integrity test...")
        frame = np.full((720, 1280, 3), 120, dtype=np.uint8)
    else:
        print(f"[2/4] Loading real test frame from {test_image_path}...")
        frame = cv2.imread(test_image_path)

    # Warm-up pass (allocates memory in ONNX runtime)
    _ = detector.detect(frame)

    print("[3/4] Benchmarking CPU inference latency over 30 iterations...")
    latencies = []
    for _ in range(30):
        t0 = time.perf_counter()
        detections = detector.detect(frame)
        latencies.append((time.perf_counter() - t0) * 1000)

    avg_ms = np.mean(latencies)
    p95_ms = np.percentile(latencies, 95)
    fps = 1000.0 / avg_ms

    print(f"      Average Latency: {avg_ms:.2f} ms")
    print(f"      P95 Latency:     {p95_ms:.2f} ms")
    print(f"      Calculated FPS:  {fps:.1f} FPS (Target: >= 18 FPS)")

    print(f"[4/4] Detections count: {len(detections)}")
    for d in detections:
        box = [int(v) for v in d['bbox']]
        print(f"      -> Class: {d['class_name']:<8} | Conf: {d['confidence']*100:.1f}% | BBox: {box}")

        # Draw on frame to visually inspect
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{d['class_name']} {d['confidence']:.2f}",
            (box[0], max(20, box[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    output_path = "output_detection_test.jpg"
    cv2.imwrite(output_path, frame)
    print(f"\n[PASS] Output preview written to {output_path}")

if __name__ == "__main__":
    test_module_1()