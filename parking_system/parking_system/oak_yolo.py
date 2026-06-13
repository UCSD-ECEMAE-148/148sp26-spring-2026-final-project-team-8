import base64
import os
import threading
import cv2
import depthai as dai
import numpy as np
import requests
import time

ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "8t6NkVHN7HBtXcxMJYPH")
WORKFLOW_URL = (
    "https://serverless.roboflow.com"
    "/henrys-workspace-dysns/workflows/small-object-detection-sahi-2"
)
ROBOFLOW_INTERVAL = 1.0  # seconds between API calls
INFER_TIMEOUT    = 10
INFER_RETRIES    = 2

CAM_SIZE = (640, 480)
BAND_CENTER = 180   # vertical center of crop in camera pixels (0=top, 480=bottom)
BAND_HEIGHT = 180   # height of crop in camera pixels
EXPOSURE_US = 5000
ISO = 300
WHITE_BALANCE_K = 6700

COLORS = {"Blue tape lines": (0, 200, 255), "empty parking space between tape lines": (0, 255, 100)}

_rf_lock = threading.Lock()
_rf_detections = []
_rf_last_call = 0.0


def roboflow_infer(frame):
    """Call SAHI workflow and return detections scaled to frame pixel coords."""
    fh, fw = frame.shape[:2]
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    b64 = base64.b64encode(buf).decode()

    payload = {
        "api_key": ROBOFLOW_API_KEY,
        "inputs": {"image": {"type": "base64", "value": b64}},
    }

    last_exc = None
    for attempt in range(INFER_RETRIES + 1):
        try:
            resp = requests.post(WORKFLOW_URL, json=payload, timeout=INFER_TIMEOUT)
            resp.raise_for_status()
            break
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < INFER_RETRIES:
                time.sleep(0.5 * (2 ** attempt))
    else:
        print(f"\n[Roboflow] failed after {INFER_RETRIES+1} attempts: {last_exc}")
        return []

    try:
        pred_block = resp.json()["outputs"][0]["predictions"]
        rf_w = pred_block["image"]["width"]
        rf_h = pred_block["image"]["height"]
        raw   = pred_block["predictions"]
    except (KeyError, IndexError, TypeError) as exc:
        print(f"\n[Roboflow] unexpected response shape: {exc}")
        return []

    # Scale from Roboflow coordinate space → display frame
    sx, sy = fw / rf_w, fh / rf_h
    results = []
    for p in raw:
        x1 = int((p["x"] - p["width"]  / 2) * sx)
        y1 = int((p["y"] - p["height"] / 2) * sy)
        x2 = int((p["x"] + p["width"]  / 2) * sx)
        y2 = int((p["y"] + p["height"] / 2) * sy)
        results.append((x1, y1, x2, y2, float(p["confidence"]), str(p["class"])))
    return results


def maybe_call_roboflow(frame):
    global _rf_last_call
    now = time.time()
    if now - _rf_last_call < ROBOFLOW_INTERVAL:
        return
    _rf_last_call = now
    frame_copy = frame.copy()

    def _worker():
        results = roboflow_infer(frame_copy)
        with _rf_lock:
            _rf_detections.clear()
            _rf_detections.extend(results)

    threading.Thread(target=_worker, daemon=True).start()


def run_pipeline():
    _crop_y = max(0, min(BAND_CENTER - BAND_HEIGHT // 2, CAM_SIZE[1] - 1))
    _crop_h = min(BAND_HEIGHT, CAM_SIZE[1] - _crop_y)
    print(f"Band crop: y={_crop_y}  h={_crop_h}")

    with dai.Pipeline() as pipeline:
        cam = pipeline.create(dai.node.Camera)
        cam.build(dai.CameraBoardSocket.CAM_A)
        cam.initialControl.setManualExposure(EXPOSURE_US, ISO)
        cam.initialControl.setManualWhiteBalance(WHITE_BALANCE_K)

        crop_manip = pipeline.create(dai.node.ImageManip)
        crop_manip.initialConfig.addCrop(0, _crop_y, CAM_SIZE[0], _crop_h)
        crop_manip.initialConfig.setOutputSize(CAM_SIZE[0], _crop_h)
        crop_manip.setMaxOutputFrameSize(CAM_SIZE[0] * _crop_h * 3)

        cam_out = cam.requestOutput(CAM_SIZE, dai.ImgFrame.Type.BGR888p, fps=5)
        cam_out.link(crop_manip.inputImage)

        frame_queue = crop_manip.out.createOutputQueue(maxSize=1, blocking=False)

        pipeline.start()
        print("Running — press Q to quit")

        while pipeline.isRunning():
            frame_msg = frame_queue.tryGet()

            if frame_msg is None:
                if cv2.waitKey(1) == ord("q"):
                    return True
                continue

            frame = frame_msg.getCvFrame()
            maybe_call_roboflow(frame)

            with _rf_lock:
                for (x1, y1, x2, y2, conf, label) in _rf_detections:
                    color = COLORS.get(label, (255, 255, 255))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label} {conf:.2f}",
                                (x1, max(y1 - 6, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            cv2.imshow("OAK-D + Roboflow", frame)
            if cv2.waitKey(1) == ord("q"):
                return True

    return False


while True:
    try:
        if run_pipeline():
            break
    except Exception as e:
        print(f"\nPipeline error: {e}")
    print("Restarting in 2s...")
    time.sleep(2)

cv2.destroyAllWindows()
