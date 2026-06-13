"""
Smoke test: send latest.jpg to the SAHI workflow and assert expected output keys.
Run with: python3 smoke_test_roboflow.py
Requires ROBOFLOW_API_KEY env var (or falls back to hardcoded key in oak_yolo.py).
"""
import base64, os, sys
import requests

API_KEY = os.environ.get("ROBOFLOW_API_KEY", "8t6NkVHN7HBtXcxMJYPH")
WORKFLOW_URL = (
    "https://serverless.roboflow.com"
    "/henrys-workspace-dysns/workflows/small-object-detection-sahi-2"
)
IMAGE_PATH = os.path.join(os.path.dirname(__file__), "latest.jpg")

with open(IMAGE_PATH, "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

resp = requests.post(
    WORKFLOW_URL,
    json={"api_key": API_KEY, "inputs": {"image": {"type": "base64", "value": b64}}},
    timeout=30,
)
resp.raise_for_status()
data = resp.json()

# Assert structure
assert "outputs" in data,                          "missing 'outputs' key"
assert len(data["outputs"]) > 0,                   "'outputs' is empty"
out = data["outputs"][0]
assert "predictions" in out,                        "missing 'predictions' in output"
pred_block = out["predictions"]
assert "image" in pred_block,                       "missing 'image' in predictions"
assert "predictions" in pred_block,                 "missing nested 'predictions' list"
assert "width"  in pred_block["image"],             "missing image width"
assert "height" in pred_block["image"],             "missing image height"

preds = pred_block["predictions"]
print(f"OK — {len(preds)} detection(s) returned")
for p in preds:
    print(f"  {p['class']:45s}  conf={p['confidence']:.3f}  "
          f"box=({p['x']:.0f},{p['y']:.0f},{p['width']:.0f}x{p['height']:.0f})")
