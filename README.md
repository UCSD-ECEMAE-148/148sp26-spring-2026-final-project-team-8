# UCSD ECEMAE148 Team 8 Final Project — Autonomous Parking System

<!-- PROJECT LOGO -->
<div align="center">
  <h1>Autonomous Parking RoboCar</h1>
  <h3>ECE MAE 148 Final Project — Team 8, Spring 2026</h3>
  <!-- Add a photo of your robot here -->
  <!-- <img src="robot.jpg" width="500"> -->
</div>

<img width="3000" height="4000" alt="image" src="https://github.com/user-attachments/assets/57fa4e53-402e-4ad7-bad1-075ba5f41f5e" />


---

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#team-members">Team Members</a></li>
    <li><a href="#final-project">Final Project</a>
      <ul>
        <li><a href="#original-goals">Original Goals</a></li>
        <li><a href="#goals-we-met">Goals We Met</a></li>
        <li><a href="#videos">Videos of Our Robot in Action</a></li>
        <li><a href="#if-we-had-another-week">If We Had Another Week...</a></li>
      </ul>
    </li>
    <li><a href="#system-overview">System Overview</a></li>
    <li><a href="#robot-design">Robot Design</a>
      <ul>
        <li><a href="#electronic-hardware">Electronic Hardware</a></li>
        <li><a href="#software-architecture">Software Architecture</a></li>
      </ul>
    </li>
    <li><a href="#how-to-run">How to Run</a></li>
    <li><a href="#docker-environment">Docker Environment</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
    <li><a href="#authors">Authors</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

---

## Team Members


| Name | Major |
|------|-------|
| Henry Tiet | Electrical & Computer Engineering |
| Geonmoo Lee | Mechanical & Aerospace Engineering |
| Christopher Wong | Mechanical & Aerospace Engineering |

---

## Final Project

### Original Goals

Our goal was to build a fully-autonomous parallel parking system for the UCSD RoboCar. The robot should:
1. Drive forward past parking spaces while scanning for an open spot.
2. Detect available spaces using computer vision (OAK-D Lite + Roboflow).
3. Classify the open space as left or right of the vehicle.
4. Execute a multi-step parallel parking maneuver using odometry feedback from the VESC motor controller.

### Goals We Met

- **Parking space detection** — We trained a custom Roboflow model to detect two classes: *blue tape lines* (the space boundaries) and *empty parking space between tape lines*. The OAK-D Lite camera crops a horizontal band of the frame to focus on the ground, sends it to the Roboflow serverless inference API, and classifies the result as `"left"`, `"right"`, or `"none"` based on which side of the frame an empty space appears.

- **Autonomous state machine** — A `ParkingControllerNode` drives the full pipeline through states: `ENABLING → SCANNING → ADVANCING → PARKING_LEFT / PARKING_RIGHT → PARKED`. If no space is found the car advances and scans again.

- **Odometry-based parking maneuvers** — The `parking_node` executes two configurable parallel-parking sequences (left and right) using VESC odometry with a speed profile (ramp-down near targets) and per-segment timeout safety fallbacks.

- **HSV blue-tape detection & BEV** — As a secondary perception layer, `parking_node` detects the blue tape lines via tunable HSV thresholds and applies a Bird's Eye View (BEV) perspective transform for accurate lane/space geometry.

- **Debug tooling** — Scan images (pre- and post-inference) are saved to `/tmp/` inside the container so they can be inspected without a display (`docker cp robocar_team8:/tmp/parking_scan_result.jpg .`).

### Videos of Our Robot in Action

https://youtube.com/shorts/1gtUK5bpeUI

https://youtube.com/shorts/GxLNIVexZjE

https://youtube.com/shorts/YCyL7NqtCjI

### If We Had Another Week...

#### Stretch Goal 1 — Full integration test
Run the complete system outdoors in the EBU courtyard, fine-tune the odometry segment distances in `parking_params.yaml` to real-world geometry, and record a clean end-to-end demo.

#### Stretch Goal 2 — Lidar-assisted space confirmation
Use the LD06 lidar to measure the physical depth of a detected space before committing to the parking maneuver, reducing false-positive detections from the camera alone.

---

## System Overview

```
OAK-D Lite Camera
      │
      ▼
OakDetectionNode ──── Roboflow Inference API
      │  (publishes /parking_status)
      │  (serves /request_scan service)
      ▼
ParkingControllerNode  (state machine)
      │  (publishes /parking_system/key)
      ▼
ParkingNode  ──── VESC Odometry (/sensors/core, /odom)
      │
      ▼
AckermannDriveStamped → /drive → VESC motor controller
```

---

## Robot Design

### Electronic Hardware

| Component | Purpose |
|-----------|---------|
| Raspberry Pi 5 | Main compute |
| OAK-D Lite | RGB camera + spatial AI |
| VESC 6 | Motor controller + odometry |
| RC car chassis | Ackermann drive platform |

### Software Architecture

The project is a single ROS2 Python package (`parking_system`) with the following nodes:

| Node | File | Role |
|------|------|------|
| `oak_detection_node` | `oak_detection_node.py` | Camera capture + Roboflow inference + space classification |
| `parking_controller_node` | `parking_controller_node.py` | Autonomous state machine |
| `parking_node` | `parking_node.py` | Low-level motion control, HSV tuning, BEV, odometry sequences |
| `rc_keyboard_node` | `rc_keyboard_node.py` | Manual keyboard / RC override |

**Launch files:**

| Launch file | Use case |
|-------------|----------|
| `parking_auto.launch.py` | Full autonomous run (OAK + controller + parking node) |
| `parking_with_odom.launch.py` | Parking node + VESC odom only (no camera) |
| `parking_with_camera.launch.py` | Parking node + camera feed only |
| `parking_with_rc.launch.py` | Manual RC / keyboard control |
| `parking_with_webcam.launch.py` | Parking node with USB webcam fallback |
| `parking_motor_calibration.launch.py` | Motor / steering calibration GUI |

---

## How to Run

### Prerequisites

- UCSD Robocar Docker environment (see [Docker Environment](#docker-environment) below)
- A [Roboflow](https://roboflow.com) API key (the repo ships with a default key; swap it via the `ROBOFLOW_API_KEY` environment variable)
- Blue tape parking space markers on the floor

### Step 1 — Enter the Docker container

```bash
docker start <your_container_name>
docker exec -it <your_container_name> bash
source_ros2
```

### Step 2 — Clone and build

```bash
cd /home/projects/ros2_ws/src
git clone https://github.com/UCSD-ECEMAE-148/148sp26-spring-2026-final-project-team-8.git
cd ..
build_ros2
```

### Step 3 — (Optional) Set your Roboflow API key

```bash
export ROBOFLOW_API_KEY=your_key_here
```

### Step 4 — Run the full autonomous parking system

```bash
ros2 launch parking_system parking_auto.launch.py
```

The robot will:
1. Wait for motors to enable (~8 s)
2. Request a camera scan from the OAK-D node
3. Drive forward and re-scan until an empty space is found
4. Execute the parallel parking sequence and stop

### Tuning parameters

All motion and perception parameters are in `parking_system/config/parking_params.yaml`. Key values to tune for your space geometry:

```yaml
odom_forward_distance_m: 0.3          # how far to advance between scans
sequence_1_j_turn_1_distance_m: 0.4   # left-park J-turn distance
sequence_2_k_turn_1_distance_m: 0.4   # right-park K-turn distance
h_low / h_high: 91 / 115              # blue HSV hue range
```

### Inspecting debug images (no display required)

After a scan you can pull the annotated image off the robot:

```bash
docker cp <your_container_name>:/tmp/parking_scan_result.jpg .
```

### Demo

<!-- Add a YouTube link here once available -->
*Demo video coming soon.*

---

## Docker Environment

This repo includes a `Dockerfile` that extends the official UCSD Robocar image (`ghcr.io/ucsd-ecemae-148/ucsd_robocar:stable`, ROS Jazzy / Ubuntu 24.04) and builds the `parking_system` package inside it.

### Build the image

```bash
git clone https://github.com/UCSD-ECEMAE-148/148sp26-spring-2026-final-project-team-8.git
cd 148sp26-spring-2026-final-project-team-8
docker build -t parking_system .
```

### Run the container

```bash
docker run -it --privileged --network host \
  --name robocar_team8 \
  -v /home/projects:/home/projects \
  parking_system bash
```

`--privileged` is required for USB access to the OAK-D camera and VESC.

### Inside the container

```bash
source /opt/ros/jazzy/setup.bash
source /home/projects/ros2_ws/install/setup.bash
ros2 launch parking_system parking_auto.launch.py
```

---

## Acknowledgments

Big thanks to Professor Jack Silberman and our TAs (Jose and Winston) for the guidance and support throughout the quarter. The base robot platform and Docker environment come from [ucsd_robocar_hub2](https://gitlab.com/ucsd_robocar2/ucsd_robocar_hub2).

---

## Authors

Henry Tiet, Geonmoo Lee, Christopher Wong

---

## Contact

* Henry Tiet — henrytiet@gmail.com
* Geonmoo Lee — *(add email)*
* Chris Wong — christopherwong747@gmail.com
