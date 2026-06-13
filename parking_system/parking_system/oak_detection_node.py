import base64
import os
import threading
import time

import cv2
import depthai as dai
import requests
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "8t6NkVHN7HBtXcxMJYPH")
WORKFLOW_URL = (
    "https://serverless.roboflow.com"
    "/henrys-workspace-dysns/workflows/small-object-detection-sahi-2"
)
INFER_TIMEOUT = 15
INFER_RETRIES = 2

CAM_SIZE = (640, 480)
BAND_CENTER = 180
BAND_HEIGHT = 180
EXPOSURE_US = 5000
ISO = 300
WHITE_BALANCE_K = 6700

EMPTY_SPACE_CLASS = "empty parking space between tape lines"
COLORS = {
    "Blue tape lines": (0, 200, 255),
    EMPTY_SPACE_CLASS: (0, 255, 100),
}
MID_X = CAM_SIZE[0] // 2  # 320

# Save debug images to disk — no X11/display required
SCAN_SENT_PATH   = "/tmp/parking_scan_sent.jpg"
SCAN_RESULT_PATH = "/tmp/parking_scan_result.jpg"


class OakDetectionNode(Node):
    def __init__(self):
        super().__init__("oak_detection_node")

        self._frame_lock = threading.Lock()
        self._latest_frame = None
        self._first_frame_time = None   # set when camera delivers its first frame
        self._running = True

        self._pub = self.create_publisher(String, "/parking_status", 10)
        self._srv = self.create_service(Trigger, "/request_scan", self._scan_callback)

        t = threading.Thread(target=self._camera_loop, daemon=True)
        t.start()

        self.get_logger().info("OakDetectionNode ready — camera starting")
        self.get_logger().info(
            f"Debug images will be written to {SCAN_SENT_PATH} and {SCAN_RESULT_PATH}"
        )

    # ── camera ────────────────────────────────────────────────────────────────

    def _camera_loop(self):
        crop_y = max(0, BAND_CENTER - BAND_HEIGHT // 2)
        crop_h = min(BAND_HEIGHT, CAM_SIZE[1] - crop_y)
        while self._running:
            try:
                self._run_pipeline(crop_y, crop_h)
            except Exception as exc:
                self.get_logger().error(f"Camera pipeline error: {exc}")
            if self._running:
                time.sleep(2.0)

    def _run_pipeline(self, crop_y, crop_h):
        with dai.Pipeline() as pipeline:
            cam = pipeline.create(dai.node.Camera)
            cam.build(dai.CameraBoardSocket.CAM_A)
            # cam.initialControl.setManualExposure(EXPOSURE_US, ISO)
            # cam.initialControl.setManualWhiteBalance(WHITE_BALANCE_K)

            manip = pipeline.create(dai.node.ImageManip)
            manip.initialConfig.addCrop(0, crop_y, CAM_SIZE[0], crop_h)
            manip.initialConfig.setOutputSize(CAM_SIZE[0], crop_h)
            manip.setMaxOutputFrameSize(CAM_SIZE[0] * crop_h * 3)

            cam_out = cam.requestOutput(CAM_SIZE, dai.ImgFrame.Type.BGR888p, fps=5)
            cam_out.link(manip.inputImage)

            q = manip.out.createOutputQueue(maxSize=1, blocking=False)
            pipeline.start()

            while self._running and pipeline.isRunning():
                msg = q.tryGet()
                if msg is not None:
                    with self._frame_lock:
                        self._latest_frame = msg.getCvFrame()
                        if self._first_frame_time is None:
                            self._first_frame_time = time.time()
                time.sleep(0.05)

    # ── service callback ───────────────────────────────────────────────────────

    def _scan_callback(self, request, response):
        try:
            # Wait for the camera to deliver its first frame (up to 30s)
            deadline = time.time() + 30.0
            frame = None
            while time.time() < deadline:
                with self._frame_lock:
                    if self._latest_frame is not None:
                        frame = self._latest_frame.copy()
                if frame is not None:
                    break
                self.get_logger().info(
                    "Waiting for first camera frame...", throttle_duration_sec=3
                )
                time.sleep(0.5)

            if frame is None:
                self.get_logger().error("Camera did not produce a frame within 30s")
                response.success = False
                response.message = "none"
                return response

            # Wait for the camera to finish auto-exposing after the first frame
            WARMUP_SEC = 5.0
            warmup_remaining = (self._first_frame_time + WARMUP_SEC) - time.time()
            if warmup_remaining > 0:
                self.get_logger().info(f"Camera warming up — waiting {warmup_remaining:.1f}s for auto-exposure to settle")
                time.sleep(warmup_remaining)
                with self._frame_lock:
                    frame = self._latest_frame.copy()

            # Save the frame being sent
            self._save_frame(frame, [], "Sending to Roboflow...", SCAN_SENT_PATH)

            detections = self._roboflow_infer(frame)
            result = self._classify(detections)

            # Save the annotated result
            self._save_frame(frame, detections, f"Result: {result.upper()}", SCAN_RESULT_PATH)

            out = String()
            out.data = result
            self._pub.publish(out)

            response.success = True
            response.message = result
            empty = [d for d in detections if d[5] == EMPTY_SPACE_CLASS]
            self.get_logger().info(
                f"Scan: {result}  ({len(empty)} empty-space, {len(detections)} total detections)"
            )
            self.get_logger().info(f"Images saved — inspect with: docker cp robocar_team8:{SCAN_RESULT_PATH} .")
        except Exception as exc:
            self.get_logger().error(f"Scan callback error: {exc}")
            response.success = False
            response.message = "none"
        return response

    # ── Roboflow ───────────────────────────────────────────────────────────────

    def _roboflow_infer(self, frame):
        """Returns all detections as (x1,y1,x2,y2,confidence,class_name) in frame pixels."""
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
            self.get_logger().error(f"Roboflow failed after retries: {last_exc}")
            return []

        try:
            pred_block = resp.json()["outputs"][0]["predictions"]
            rf_w = pred_block["image"]["width"]
            rf_h = pred_block["image"]["height"]
            raw = pred_block["predictions"]
            if not rf_w or not rf_h:
                self.get_logger().error(
                    f"Roboflow returned null image dimensions: {pred_block['image']}"
                )
                return []
            sx, sy = fw / rf_w, fh / rf_h
            results = []
            for p in raw:
                x1 = int((p["x"] - p["width"] / 2) * sx)
                y1 = int((p["y"] - p["height"] / 2) * sy)
                x2 = int((p["x"] + p["width"] / 2) * sx)
                y2 = int((p["y"] + p["height"] / 2) * sy)
                results.append((x1, y1, x2, y2, float(p["confidence"]), str(p["class"])))
            return results
        except (KeyError, IndexError, TypeError) as exc:
            self.get_logger().error(f"Roboflow response parse error: {exc}")
            return []

    # ── classify ───────────────────────────────────────────────────────────────

    def _classify(self, detections):
        empty_cx = [
            ((x1 + x2) // 2, conf)
            for x1, y1, x2, y2, conf, cls in detections
            if cls == EMPTY_SPACE_CLASS
        ]
        left_conf  = max((c for cx, c in empty_cx if cx <  MID_X), default=0.0)
        right_conf = max((c for cx, c in empty_cx if cx >= MID_X), default=0.0)
        if left_conf == 0.0 and right_conf == 0.0:
            return "none"
        return "left" if left_conf >= right_conf else "right"

    # ── debug image ────────────────────────────────────────────────────────────

    def _save_frame(self, frame, detections, status_text, path):
        display = frame.copy()
        h, w = display.shape[:2]

        cv2.line(display, (MID_X, 0), (MID_X, h), (180, 180, 180), 1)

        for x1, y1, x2, y2, conf, label in detections:
            color = COLORS.get(label, (255, 255, 255))
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                display, f"{label[:20]} {conf:.2f}",
                (x1, max(y1 - 6, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
            )

        cv2.putText(
            display, status_text, (8, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )
        cv2.imwrite(path, display)

    # ── lifecycle ──────────────────────────────────────────────────────────────

    def destroy_node(self):
        self._running = False
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = OakDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
