from datetime import datetime
from pathlib import Path

from ackermann_msgs.msg import AckermannDriveStamped
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
import cv2
import math
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String
import time
from vesc_msgs.msg import VescStateStamped
import yaml


NODE_NAME = 'parking_node'
PACKAGE_NAME = 'parking_system'

ORIGINAL_WINDOW = 'Original Camera'
BLUE_MASK_WINDOW = 'Blue Mask'
CLEANED_MASK_WINDOW = 'Cleaned Mask'
CANNY_EDGES_WINDOW = 'Canny Edges'
DEBUG_OVERLAY_WINDOW = 'Debug Overlay'
BEV_IMAGE_WINDOW = 'BEV Image'
BEV_MASK_WINDOW = 'BEV Mask'
BEV_LINE_OVERLAY_WINDOW = 'BEV Line Overlay'
APPROACH_ROI_MASK_WINDOW = 'Approach ROI Mask'
TRACKBAR_WINDOW = 'Blue HSV Tuning'
MOTOR_TRACKBAR_WINDOW = 'Motor Calibration Tuning'

TUNABLE_PARAMETERS = (
    'h_low',
    'h_high',
    's_low',
    's_high',
    'v_low',
    'v_high',
    'morphology_kernel_size',
    'minimum_contour_area',
    'canny_low',
    'canny_high',
    'hough_threshold',
    'min_line_length',
    'max_line_gap',
    'bev_src_p1_x',
    'bev_src_p1_y',
    'bev_src_p2_x',
    'bev_src_p2_y',
    'bev_src_p3_x',
    'bev_src_p3_y',
    'bev_src_p4_x',
    'bev_src_p4_y',
)

TRACKBAR_SPECS = (
    ('H low', 'h_low', 179),
    ('H high', 'h_high', 179),
    ('S low', 's_low', 255),
    ('S high', 's_high', 255),
    ('V low', 'v_low', 255),
    ('V high', 'v_high', 255),
    ('morphology kernel size', 'morphology_kernel_size', 31),
    ('minimum contour area', 'minimum_contour_area', 50000),
    ('Canny low', 'canny_low', 255),
    ('Canny high', 'canny_high', 255),
    ('Hough threshold', 'hough_threshold', 300),
    ('minimum line length', 'min_line_length', 500),
    ('maximum line gap', 'max_line_gap', 200),
    ('BEV p1 x', 'bev_src_p1_x', 2000),
    ('BEV p1 y', 'bev_src_p1_y', 2000),
    ('BEV p2 x', 'bev_src_p2_x', 2000),
    ('BEV p2 y', 'bev_src_p2_y', 2000),
    ('BEV p3 x', 'bev_src_p3_x', 2000),
    ('BEV p3 y', 'bev_src_p3_y', 2000),
    ('BEV p4 x', 'bev_src_p4_x', 2000),
    ('BEV p4 y', 'bev_src_p4_y', 2000),
)

MOTOR_TRACKBAR_SPECS = (
    ('linear speed x1000', 'test_linear_speed', 1000),
    ('backward speed x1000', 'test_backward_speed', 1000),
    ('left angular x1000', 'test_angular_left', 2000),
    ('right angular x1000', 'test_angular_right', 2000),
    ('pulse duration ms', 'test_pulse_duration_sec', 5000),
)


def package_root_from_module():
    return Path(__file__).resolve().parents[1]


def default_config_file():
    source_config = package_root_from_module() / 'config' / 'parking_params.yaml'
    if source_config.exists():
        return source_config

    try:
        share_dir = Path(get_package_share_directory(PACKAGE_NAME))
        share_config = share_dir / 'config' / 'parking_params.yaml'
        if share_config.exists():
            return share_config
    except Exception:
        pass

    return source_config


def default_debug_image_dir():
    source_debug_dir = package_root_from_module() / 'debug_images'
    if source_debug_dir.exists():
        return source_debug_dir

    try:
        return Path(get_package_share_directory(PACKAGE_NAME)) / 'debug_images'
    except Exception:
        return source_debug_dir


class ParkingNode(Node):
    def __init__(self):
        super().__init__(NODE_NAME)
        self.bridge = CvBridge()
        self.queue_size = 10
        self.shutdown_requested = False
        self.gui_available = False
        self.pending_autosave = False
        self.last_trackbar_change_time = 0.0
        self.last_ros_image_time = None
        self.last_opencv_frame_time = None
        self.ros_frame_count = 0
        self.opencv_frame_count = 0
        self.video_capture = None
        self.active_opencv_source = None
        self.last_opencv_open_attempt = 0.0
        self.last_camera_warning_time = 0.0
        self.opencv_fallback_logged = False

        self._declare_ros_parameters()
        self._load_ros_parameters()

        self.latest_original = None
        self.latest_mask = None
        self.latest_cleaned_mask = None
        self.latest_edges = None
        self.latest_overlay = None
        self.latest_bev_image = None
        self.latest_bev_mask = None
        self.latest_bev_line_overlay = None
        self.latest_approach_roi_mask = None
        self.current_approach_component_count = 0
        self.current_approach_state = 'APPROACH'
        self.approach_stable_frame_count = 0
        self.current_approach_components = []
        self.active_motor_pulse = None
        self.motor_pulse_start_time = None
        self.active_motor_pulse_duration_sec = None
        self.active_motor_sequence = None
        self.motor_sequence_steps = []
        self.motor_sequence_step_index = 0
        self.latest_odom_pose = None
        self.latest_odom_time = None
        self.odom_frame_count = 0
        self.latest_vesc_state_time = None
        self.latest_vesc_state_monotonic = None
        self.latest_vesc_speed_mps = 0.0
        self.vesc_state_frame_count = 0
        self.integrated_vesc_distance_m = 0.0
        self.active_odom_motion = None
        self.odom_motion_start_pose = None
        self.odom_motion_start_vesc_distance_m = 0.0
        self.odom_motion_feedback_source = 'none'
        self.odom_motion_start_time = None
        self.odom_motion_target_distance_m = 0.0
        self.odom_motion_timeout_sec = 0.0
        self.last_motor_command_preview = {
            'linear_x': 0.0,
            'angular_z': 0.0,
        }
        self.last_motor_command_status = 'none'
        self.rc_linear_x = 0.0
        self.rc_angular_z = 0.0

        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            self.cmd_vel_topic,
            self.queue_size,
        )
        self.ackermann_publisher = self.create_publisher(
            AckermannDriveStamped,
            self.ackermann_topic,
            self.queue_size,
        )
        self.image_subscription = None
        if self.perception_enabled:
            self.image_subscription = self.create_subscription(
                Image,
                self.camera_topic,
                self.image_callback,
                qos_profile_sensor_data,
            )
        self.odom_subscription = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            self.queue_size,
        )
        self.odom_subscription
        self.vesc_state_subscription = self.create_subscription(
            VescStateStamped,
            self.vesc_state_topic,
            self.vesc_state_callback,
            self.queue_size,
        )
        self.vesc_state_subscription
        self.keyboard_subscription = None
        if self.keyboard_topic_control_enabled:
            self.keyboard_subscription = self.create_subscription(
                String,
                self.keyboard_topic,
                self.keyboard_topic_callback,
                self.queue_size,
            )

        if self.enable_gui:
            self.get_logger().info(
                'Creating OpenCV GUI windows. '
                f'motor_calibration_gui_only={self.motor_calibration_gui_only}'
            )
            self._setup_gui()
            self.get_logger().info(
                f'OpenCV GUI available: {self.gui_available}'
            )
        if self.gui_available:
            self.gui_timer = self.create_timer(
                self.gui_timer_period,
                self.gui_timer_callback,
            )
        if self.perception_enabled and self.use_opencv_camera_fallback:
            self.opencv_camera_timer = self.create_timer(
                self.opencv_timer_period,
                self.opencv_camera_timer_callback,
            )

        self.safety_timer = self.create_timer(
            self.safety_timer_period,
            self.safety_timer_callback,
        )
        if self.perception_enabled:
            self.camera_status_timer = self.create_timer(
                2.0,
                self.camera_status_timer_callback,
            )

        self.get_logger().info(
            f'\n camera_topic: {self.camera_topic}'
            f'\n cmd_vel_topic: {self.cmd_vel_topic}'
            f'\n ackermann_topic: {self.ackermann_topic}'
            f'\n publish_ackermann: {self.publish_ackermann}'
            f'\n odom_topic: {self.odom_topic}'
            f'\n odom_control_enabled: {self.odom_control_enabled}'
            f'\n vesc_state_topic: {self.vesc_state_topic}'
            f'\n use_vesc_state_distance_fallback: '
            f'{self.use_vesc_state_distance_fallback}'
            f'\n safety_timer_period: {self.safety_timer_period}'
            f'\n perception_enabled: {self.perception_enabled}'
            f'\n enable_gui: {self.enable_gui}'
            f'\n keyboard_topic_control_enabled: '
            f'{self.keyboard_topic_control_enabled}'
            f'\n keyboard_topic: {self.keyboard_topic}'
            f'\n auto_save_on_trackbar_change: '
            f'{self.auto_save_on_trackbar_change}'
            f'\n use_opencv_camera_fallback: '
            f'{self.use_opencv_camera_fallback}'
            f'\n config_file: {self.config_file}'
            f'\n debug_image_dir: {self.debug_image_dir}'
        )

    def _declare_ros_parameters(self):
        self.declare_parameters(
            namespace='',
            parameters=[
                ('camera_topic', '/camera/color/image_raw'),
                ('cmd_vel_topic', '/cmd_vel'),
                ('ackermann_topic', '/drive'),
                ('publish_ackermann', True),
                ('ackermann_steering_scale', 0.35),
                ('odom_topic', '/odom'),
                ('odom_control_enabled', False),
                ('odom_timeout_sec', 1.0),
                ('odom_forward_distance_m', 0.10),
                ('odom_backward_distance_m', 0.10),
                ('odom_turn_distance_m', 0.10),
                ('straight_correction_angular_z', 0.10),
                ('sequence_1_forward_start_distance_m', 0.10),
                ('sequence_1_j_turn_1_distance_m', 0.40),
                ('sequence_1_backward_distance_m', 0.20),
                ('sequence_1_j_turn_2_distance_m', 0.30),
                ('sequence_1_backward_2_distance_m', 0.20),
                ('sequence_1_j_turn_3_distance_m', 0.20),
                ('sequence_2_forward_start_distance_m', 0.10),
                ('sequence_2_k_turn_1_distance_m', 0.40),
                ('sequence_2_backward_distance_m', 0.20),
                ('sequence_2_k_turn_2_distance_m', 0.25),
                ('sequence_2_backward_2_distance_m', 0.20),
                ('sequence_2_k_turn_3_distance_m', 0.20),
                ('odom_segment_timeout_sec', 4.0),
                ('odom_stop_buffer_m', 0.04),
                ('odom_speed_profile_enabled', True),
                ('odom_slowdown_distance_m', 0.08),
                ('odom_min_speed_scale', 0.75),
                ('odom_min_linear_speed', 0.22),
                ('use_vesc_state_distance_fallback', True),
                ('vesc_state_topic', '/sensors/core'),
                ('vesc_state_timeout_sec', 1.0),
                ('vesc_speed_to_erpm_gain', 4921.82),
                ('vesc_speed_to_erpm_offset', 93.59),
                ('safety_timer_period', 0.1),
                ('perception_enabled', True),
                ('keyboard_topic_control_enabled', False),
                ('keyboard_topic', '/parking_system/key'),
                ('enable_gui', True),
                ('motor_calibration_gui_only', False),
                ('gui_timer_period', 0.03),
                ('auto_save_on_trackbar_change', True),
                ('auto_save_debounce_sec', 0.5),
                ('use_opencv_camera_fallback', True),
                ('opencv_camera_source', ''),
                ('opencv_camera_search_indices', '0,1,2,3,4'),
                ('opencv_camera_index', 0),
                ('opencv_camera_width', 0),
                ('opencv_camera_height', 0),
                ('opencv_timer_period', 0.03),
                ('ros_image_timeout', 2.0),
                ('config_file', str(default_config_file())),
                ('debug_image_dir', str(default_debug_image_dir())),
                ('h_low', 90),
                ('h_high', 130),
                ('s_low', 80),
                ('s_high', 255),
                ('v_low', 40),
                ('v_high', 255),
                ('morphology_kernel_size', 5),
                ('minimum_contour_area', 300),
                ('canny_low', 50),
                ('canny_high', 150),
                ('hough_threshold', 30),
                ('min_line_length', 30),
                ('max_line_gap', 10),
                ('bev_enabled', True),
                ('bev_width', 600),
                ('bev_height', 800),
                ('bev_src_p1_x', 160),
                ('bev_src_p1_y', 220),
                ('bev_src_p2_x', 480),
                ('bev_src_p2_y', 220),
                ('bev_src_p3_x', 620),
                ('bev_src_p3_y', 470),
                ('bev_src_p4_x', 20),
                ('bev_src_p4_y', 470),
                ('approach_trigger_enabled', True),
                ('approach_roi_enabled', False),
                ('approach_roi_x_min', 0),
                ('approach_roi_y_min', 0),
                ('approach_roi_x_max', 1280),
                ('approach_roi_y_max', 800),
                ('approach_min_component_area', 300),
                ('approach_max_component_area', 100000),
                ('approach_target_component_count', 4),
                ('approach_count_tolerance', 1),
                ('approach_required_stable_frames', 5),
                ('motor_calibration_enabled', False),
                ('motion_enabled', False),
                ('rc_control_enabled', False),
                ('rc_linear_step', 0.05),
                ('rc_angular_step', 0.10),
                ('rc_max_linear_speed', 0.50),
                ('rc_max_angular_z', 1.30),
                ('test_linear_speed', 0.25),
                ('test_backward_speed', 0.25),
                ('test_angular_left', 0.20),
                ('test_angular_right', -0.20),
                ('test_pulse_duration_sec', 1.0),
            ],
        )

    def _load_ros_parameters(self):
        self.camera_topic = self.get_parameter('camera_topic').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.ackermann_topic = self.get_parameter('ackermann_topic').value
        self.publish_ackermann = bool(
            self.get_parameter('publish_ackermann').value
        )
        self.ackermann_steering_scale = float(
            self.get_parameter('ackermann_steering_scale').value
        )
        self.odom_topic = self.get_parameter('odom_topic').value
        self.odom_control_enabled = bool(
            self.get_parameter('odom_control_enabled').value
        )
        self.odom_timeout_sec = float(
            self.get_parameter('odom_timeout_sec').value
        )
        self.odom_forward_distance_m = float(
            self.get_parameter('odom_forward_distance_m').value
        )
        self.odom_backward_distance_m = float(
            self.get_parameter('odom_backward_distance_m').value
        )
        self.odom_turn_distance_m = float(
            self.get_parameter('odom_turn_distance_m').value
        )
        self.straight_correction_angular_z = float(
            self.get_parameter('straight_correction_angular_z').value
        )
        self.sequence_1_forward_start_distance_m = float(
            self.get_parameter('sequence_1_forward_start_distance_m').value
        )
        self.sequence_1_j_turn_1_distance_m = float(
            self.get_parameter('sequence_1_j_turn_1_distance_m').value
        )
        self.sequence_1_backward_distance_m = float(
            self.get_parameter('sequence_1_backward_distance_m').value
        )
        self.sequence_1_j_turn_2_distance_m = float(
            self.get_parameter('sequence_1_j_turn_2_distance_m').value
        )
        self.sequence_1_backward_2_distance_m = float(
            self.get_parameter('sequence_1_backward_2_distance_m').value
        )
        self.sequence_1_j_turn_3_distance_m = float(
            self.get_parameter('sequence_1_j_turn_3_distance_m').value
        )
        self.sequence_2_forward_start_distance_m = float(
            self.get_parameter('sequence_2_forward_start_distance_m').value
        )
        self.sequence_2_k_turn_1_distance_m = float(
            self.get_parameter('sequence_2_k_turn_1_distance_m').value
        )
        self.sequence_2_backward_distance_m = float(
            self.get_parameter('sequence_2_backward_distance_m').value
        )
        self.sequence_2_k_turn_2_distance_m = float(
            self.get_parameter('sequence_2_k_turn_2_distance_m').value
        )
        self.sequence_2_backward_2_distance_m = float(
            self.get_parameter('sequence_2_backward_2_distance_m').value
        )
        self.sequence_2_k_turn_3_distance_m = float(
            self.get_parameter('sequence_2_k_turn_3_distance_m').value
        )
        self.odom_segment_timeout_sec = float(
            self.get_parameter('odom_segment_timeout_sec').value
        )
        self.odom_stop_buffer_m = float(
            self.get_parameter('odom_stop_buffer_m').value
        )
        self.odom_speed_profile_enabled = bool(
            self.get_parameter('odom_speed_profile_enabled').value
        )
        self.odom_slowdown_distance_m = float(
            self.get_parameter('odom_slowdown_distance_m').value
        )
        self.odom_min_speed_scale = float(
            self.get_parameter('odom_min_speed_scale').value
        )
        self.odom_min_linear_speed = float(
            self.get_parameter('odom_min_linear_speed').value
        )
        self.use_vesc_state_distance_fallback = bool(
            self.get_parameter('use_vesc_state_distance_fallback').value
        )
        self.vesc_state_topic = self.get_parameter('vesc_state_topic').value
        self.vesc_state_timeout_sec = float(
            self.get_parameter('vesc_state_timeout_sec').value
        )
        self.vesc_speed_to_erpm_gain = float(
            self.get_parameter('vesc_speed_to_erpm_gain').value
        )
        self.vesc_speed_to_erpm_offset = float(
            self.get_parameter('vesc_speed_to_erpm_offset').value
        )
        self.safety_timer_period = float(
            self.get_parameter('safety_timer_period').value
        )
        self.perception_enabled = bool(
            self.get_parameter('perception_enabled').value
        )
        self.keyboard_topic_control_enabled = bool(
            self.get_parameter('keyboard_topic_control_enabled').value
        )
        self.keyboard_topic = str(
            self.get_parameter('keyboard_topic').value
        )
        self.enable_gui = bool(self.get_parameter('enable_gui').value)
        self.motor_calibration_gui_only = bool(
            self.get_parameter('motor_calibration_gui_only').value
        )
        self.gui_timer_period = float(
            self.get_parameter('gui_timer_period').value
        )
        self.auto_save_on_trackbar_change = bool(
            self.get_parameter('auto_save_on_trackbar_change').value
        )
        self.auto_save_debounce_sec = float(
            self.get_parameter('auto_save_debounce_sec').value
        )
        self.use_opencv_camera_fallback = bool(
            self.get_parameter('use_opencv_camera_fallback').value
        )
        self.opencv_camera_source = str(
            self.get_parameter('opencv_camera_source').value
        )
        self.opencv_camera_search_indices = str(
            self.get_parameter('opencv_camera_search_indices').value
        )
        self.opencv_camera_index = int(
            self.get_parameter('opencv_camera_index').value
        )
        self.opencv_camera_width = int(
            self.get_parameter('opencv_camera_width').value
        )
        self.opencv_camera_height = int(
            self.get_parameter('opencv_camera_height').value
        )
        self.opencv_timer_period = float(
            self.get_parameter('opencv_timer_period').value
        )
        self.ros_image_timeout = float(
            self.get_parameter('ros_image_timeout').value
        )
        self.bev_enabled = bool(self.get_parameter('bev_enabled').value)
        self.bev_width = int(self.get_parameter('bev_width').value)
        self.bev_height = int(self.get_parameter('bev_height').value)
        self.approach_trigger_enabled = bool(
            self.get_parameter('approach_trigger_enabled').value
        )
        self.approach_roi_enabled = bool(
            self.get_parameter('approach_roi_enabled').value
        )
        self.approach_roi_x_min = int(
            self.get_parameter('approach_roi_x_min').value
        )
        self.approach_roi_y_min = int(
            self.get_parameter('approach_roi_y_min').value
        )
        self.approach_roi_x_max = int(
            self.get_parameter('approach_roi_x_max').value
        )
        self.approach_roi_y_max = int(
            self.get_parameter('approach_roi_y_max').value
        )
        self.approach_min_component_area = int(
            self.get_parameter('approach_min_component_area').value
        )
        self.approach_max_component_area = int(
            self.get_parameter('approach_max_component_area').value
        )
        self.approach_target_component_count = int(
            self.get_parameter('approach_target_component_count').value
        )
        self.approach_count_tolerance = int(
            self.get_parameter('approach_count_tolerance').value
        )
        self.approach_required_stable_frames = int(
            self.get_parameter('approach_required_stable_frames').value
        )
        self.motor_calibration_enabled = bool(
            self.get_parameter('motor_calibration_enabled').value
        )
        self.motion_enabled = bool(
            self.get_parameter('motion_enabled').value
        )
        self.rc_control_enabled = bool(
            self.get_parameter('rc_control_enabled').value
        )
        self.rc_linear_step = float(
            self.get_parameter('rc_linear_step').value
        )
        self.rc_angular_step = float(
            self.get_parameter('rc_angular_step').value
        )
        self.rc_max_linear_speed = float(
            self.get_parameter('rc_max_linear_speed').value
        )
        self.rc_max_angular_z = float(
            self.get_parameter('rc_max_angular_z').value
        )
        self.test_linear_speed = float(
            self.get_parameter('test_linear_speed').value
        )
        self.test_backward_speed = float(
            self.get_parameter('test_backward_speed').value
        )
        self.test_angular_left = float(
            self.get_parameter('test_angular_left').value
        )
        self.test_angular_right = float(
            self.get_parameter('test_angular_right').value
        )
        self.test_pulse_duration_sec = float(
            self.get_parameter('test_pulse_duration_sec').value
        )
        self.config_file = Path(self.get_parameter('config_file').value)
        self.debug_image_dir = Path(
            self.get_parameter('debug_image_dir').value
        )

        if not self.debug_image_dir.is_absolute():
            self.debug_image_dir = (
                package_root_from_module() / self.debug_image_dir
            )

        self.tuning = {
            name: int(self.get_parameter(name).value)
            for name in TUNABLE_PARAMETERS
        }

    def _setup_gui(self):
        try:
            if not self.motor_calibration_gui_only:
                self.get_logger().info(
                    f'OpenCV GUI: namedWindow {TRACKBAR_WINDOW}'
                )
                cv2.namedWindow(TRACKBAR_WINDOW, cv2.WINDOW_NORMAL)
                self.get_logger().info(
                    f'OpenCV GUI: namedWindow {ORIGINAL_WINDOW}'
                )
                cv2.namedWindow(ORIGINAL_WINDOW, cv2.WINDOW_NORMAL)
                self.get_logger().info(
                    f'OpenCV GUI: namedWindow {BLUE_MASK_WINDOW}'
                )
                cv2.namedWindow(BLUE_MASK_WINDOW, cv2.WINDOW_NORMAL)
                self.get_logger().info(
                    f'OpenCV GUI: namedWindow {CLEANED_MASK_WINDOW}'
                )
                cv2.namedWindow(CLEANED_MASK_WINDOW, cv2.WINDOW_NORMAL)
                self.get_logger().info(
                    f'OpenCV GUI: namedWindow {CANNY_EDGES_WINDOW}'
                )
                cv2.namedWindow(CANNY_EDGES_WINDOW, cv2.WINDOW_NORMAL)
                self.get_logger().info(
                    f'OpenCV GUI: namedWindow {DEBUG_OVERLAY_WINDOW}'
                )
                cv2.namedWindow(DEBUG_OVERLAY_WINDOW, cv2.WINDOW_NORMAL)
                self.get_logger().info(
                    f'OpenCV GUI: namedWindow {BEV_IMAGE_WINDOW}'
                )
                cv2.namedWindow(BEV_IMAGE_WINDOW, cv2.WINDOW_NORMAL)
                self.get_logger().info(
                    f'OpenCV GUI: namedWindow {BEV_MASK_WINDOW}'
                )
                cv2.namedWindow(BEV_MASK_WINDOW, cv2.WINDOW_NORMAL)
                self.get_logger().info(
                    f'OpenCV GUI: namedWindow {BEV_LINE_OVERLAY_WINDOW}'
                )
                cv2.namedWindow(BEV_LINE_OVERLAY_WINDOW, cv2.WINDOW_NORMAL)
                self.get_logger().info(
                    f'OpenCV GUI: namedWindow {APPROACH_ROI_MASK_WINDOW}'
                )
                cv2.namedWindow(APPROACH_ROI_MASK_WINDOW, cv2.WINDOW_NORMAL)
            else:
                self.get_logger().info(
                    f'OpenCV GUI: namedWindow {DEBUG_OVERLAY_WINDOW}'
                )
                cv2.namedWindow(DEBUG_OVERLAY_WINDOW, cv2.WINDOW_NORMAL)

            self.get_logger().info(
                f'OpenCV GUI: namedWindow {MOTOR_TRACKBAR_WINDOW}'
            )
            cv2.namedWindow(MOTOR_TRACKBAR_WINDOW, cv2.WINDOW_NORMAL)

            if not self.motor_calibration_gui_only:
                for label, parameter_name, max_value in TRACKBAR_SPECS:
                    value = self._clamp(
                        self.tuning[parameter_name],
                        0,
                        max_value,
                    )
                    cv2.createTrackbar(
                        label,
                        TRACKBAR_WINDOW,
                        value,
                        max_value,
                        lambda _value: None,
                    )

            self.get_logger().info('OpenCV GUI: creating motor trackbars')
            for label, parameter_name, max_value in MOTOR_TRACKBAR_SPECS:
                value = self._motor_value_to_trackbar(
                    parameter_name,
                    max_value,
                )
                cv2.createTrackbar(
                    label,
                    MOTOR_TRACKBAR_WINDOW,
                    value,
                    max_value,
                    lambda _value: None,
                )

            self.get_logger().info('OpenCV GUI: showing initial status')
            self._show_initial_status_windows()
            self.get_logger().info('OpenCV GUI: setup complete')
            self.gui_available = True
        except cv2.error as error:
            self.gui_available = False
            self.get_logger().warning(
                'OpenCV GUI could not be created. Running headless: '
                f'{error}'
            )

    def _position_debug_windows(self):
        try:
            cv2.resizeWindow(DEBUG_OVERLAY_WINDOW, 900, 620)
            cv2.resizeWindow(MOTOR_TRACKBAR_WINDOW, 520, 220)
            cv2.moveWindow(DEBUG_OVERLAY_WINDOW, 40, 40)
            cv2.moveWindow(MOTOR_TRACKBAR_WINDOW, 980, 40)
        except cv2.error:
            pass

    def _show_initial_status_windows(self):
        status_image = self._build_status_image()
        empty_mask = np.zeros(status_image.shape[:2], dtype=np.uint8)

        cv2.imshow(DEBUG_OVERLAY_WINDOW, status_image)
        if not self.motor_calibration_gui_only:
            cv2.imshow(ORIGINAL_WINDOW, status_image)
            cv2.imshow(BLUE_MASK_WINDOW, empty_mask)
            cv2.imshow(CLEANED_MASK_WINDOW, empty_mask)
            cv2.imshow(CANNY_EDGES_WINDOW, empty_mask)
            cv2.imshow(BEV_IMAGE_WINDOW, status_image)
            cv2.imshow(BEV_MASK_WINDOW, empty_mask)
            cv2.imshow(BEV_LINE_OVERLAY_WINDOW, status_image)
            cv2.imshow(APPROACH_ROI_MASK_WINDOW, empty_mask)

        cv2.waitKey(1)

    def gui_timer_callback(self):
        if not self.gui_available:
            return

        try:
            self._read_trackbars()
            self._read_motor_trackbars()
            self._show_status_windows_if_needed()
            self._handle_keyboard(cv2.waitKey(1) & 0xFF)
            self._maybe_auto_save()
        except cv2.error as error:
            self.gui_available = False
            self.get_logger().warning(
                'OpenCV GUI failed while polling controls. '
                f'Continuing headless: {error}'
            )

    def opencv_camera_timer_callback(self):
        if not self.perception_enabled:
            return

        if not self._should_use_opencv_fallback():
            return

        capture = self._ensure_opencv_camera()
        if capture is None:
            return

        success, frame = capture.read()
        if not success:
            self._warn_throttled(
                'OpenCV fallback camera opened, but no frame was read.'
            )
            self._release_opencv_camera()
            return

        self.last_opencv_frame_time = time.monotonic()
        self.opencv_frame_count += 1
        self._process_bgr_frame(frame)

    def camera_status_timer_callback(self):
        if not self.perception_enabled:
            return

        if self.last_opencv_frame_time is not None:
            age = time.monotonic() - self.last_opencv_frame_time
            if age <= self.ros_image_timeout:
                return

        if self.last_ros_image_time is not None:
            age = time.monotonic() - self.last_ros_image_time
            if age <= self.ros_image_timeout:
                return

        if self.use_opencv_camera_fallback:
            self._warn_throttled(
                f'No recent ROS images on {self.camera_topic}. '
                'Trying OpenCV camera fallback. '
                f'{self._image_topic_summary()}'
            )
        else:
            self._warn_throttled(
                f'No images received on {self.camera_topic}. '
                'Start a ROS camera publisher or enable '
                'use_opencv_camera_fallback. '
                f'{self._image_topic_summary()}'
            )

    def image_callback(self, msg):
        if not self.perception_enabled:
            return

        try:
            bgr_image = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='bgr8',
            )
        except CvBridgeError as error:
            self.get_logger().error(f'cv_bridge conversion failed: {error}')
            return

        self.last_ros_image_time = time.monotonic()
        self.ros_frame_count += 1
        self._release_opencv_camera()
        self._process_bgr_frame(bgr_image)

    def _process_bgr_frame(self, bgr_image):
        if not self.perception_enabled:
            return

        mask, cleaned_mask, edges, contours, line_segments = (
            self.process_frame(bgr_image)
        )
        approach_roi_mask = self.build_approach_roi_mask(cleaned_mask)
        component_count, components = self.count_blue_components(cleaned_mask)
        self.current_approach_component_count = component_count
        self.current_approach_components = components
        self.update_approach_trigger(component_count)
        bev_image, bev_mask = self.compute_bird_eye_view(
            bgr_image,
            cleaned_mask,
        )
        bev_line_segments = self.detect_line_segments(bev_mask)
        bev_line_overlay = self.build_bev_line_overlay(
            bev_image,
            bev_line_segments,
        )
        overlay = self.build_debug_overlay(
            bgr_image,
            contours,
            line_segments,
        )

        self.latest_original = bgr_image.copy()
        self.latest_mask = mask.copy()
        self.latest_cleaned_mask = cleaned_mask.copy()
        self.latest_edges = edges.copy()
        self.latest_overlay = overlay.copy()
        self.latest_bev_image = bev_image.copy()
        self.latest_bev_mask = bev_mask.copy()
        self.latest_bev_line_overlay = bev_line_overlay.copy()
        self.latest_approach_roi_mask = approach_roi_mask.copy()

        self.update_extension_hooks(
            bgr_image,
            cleaned_mask,
            contours,
            line_segments,
            bev_image,
            bev_mask,
            bev_line_segments,
        )
        self._show_debug_windows(
            bgr_image,
            mask,
            cleaned_mask,
            edges,
            overlay,
            bev_image,
            bev_mask,
            bev_line_overlay,
            approach_roi_mask,
        )

    def _should_use_opencv_fallback(self):
        if self.last_ros_image_time is None:
            return True

        ros_image_age = time.monotonic() - self.last_ros_image_time
        if ros_image_age > self.ros_image_timeout:
            return True

        self._release_opencv_camera()
        return False

    def _ensure_opencv_camera(self):
        if self.video_capture is not None and self.video_capture.isOpened():
            return self.video_capture

        now = time.monotonic()
        if now - self.last_opencv_open_attempt < 2.0:
            return None

        self.last_opencv_open_attempt = now
        for source in self._opencv_sources_to_try():
            capture = self._open_video_capture(source)
            if capture is None:
                continue

            self._configure_capture(capture)
            self.video_capture = capture
            self.active_opencv_source = source
            if not self.opencv_fallback_logged:
                self.get_logger().info(
                    f'Using OpenCV camera fallback source {source}.'
                )
                self.opencv_fallback_logged = True
            return self.video_capture

        self._warn_throttled(
            'Could not open any OpenCV fallback camera source. '
            f'Tried: {self._opencv_sources_to_try()}'
        )
        return None

    def _opencv_sources_to_try(self):
        source = self.opencv_camera_source.strip()
        if source:
            return [self._parse_opencv_source(source)]

        sources = []
        for text in self.opencv_camera_search_indices.split(','):
            text = text.strip()
            if not text:
                continue
            try:
                sources.append(int(text))
            except ValueError:
                sources.append(text)

        if self.opencv_camera_index not in sources:
            sources.insert(0, self.opencv_camera_index)

        return sources

    def _parse_opencv_source(self, source):
        try:
            return int(source)
        except ValueError:
            return source

    def _open_video_capture(self, source):
        if isinstance(source, int) and not Path(f'/dev/video{source}').exists():
            return None

        if isinstance(source, int):
            capture = cv2.VideoCapture(source, cv2.CAP_V4L2)
        else:
            capture = cv2.VideoCapture(source)

        if not capture.isOpened():
            capture.release()
            return None

        return capture

    def _configure_capture(self, capture):
        if self.opencv_camera_width > 0:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.opencv_camera_width)
        if self.opencv_camera_height > 0:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.opencv_camera_height)

    def _release_opencv_camera(self):
        if self.video_capture is None:
            return

        self.video_capture.release()
        self.video_capture = None
        self.active_opencv_source = None

    def process_frame(self, bgr_image):
        mask = self.detect_blue_tape(bgr_image)
        cleaned_mask = self.clean_mask(mask)
        edges = self.detect_edges(cleaned_mask)
        line_segments = self.detect_line_segments(cleaned_mask)
        contours = self.find_tape_contours(cleaned_mask)
        return mask, cleaned_mask, edges, contours, line_segments

    def detect_blue_tape(self, bgr_image):
        hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        h_low = self._clamp(self.tuning['h_low'], 0, 179)
        h_high = self._clamp(self.tuning['h_high'], 0, 179)
        s_low = self._clamp(self.tuning['s_low'], 0, 255)
        s_high = self._clamp(self.tuning['s_high'], 0, 255)
        v_low = self._clamp(self.tuning['v_low'], 0, 255)
        v_high = self._clamp(self.tuning['v_high'], 0, 255)
        s_min, s_max = sorted((s_low, s_high))
        v_min, v_max = sorted((v_low, v_high))

        if h_low <= h_high:
            lower = np.array([h_low, s_min, v_min])
            upper = np.array([h_high, s_max, v_max])
            return cv2.inRange(hsv_image, lower, upper)

        lower_first = np.array([0, s_min, v_min])
        upper_first = np.array([h_high, s_max, v_max])
        lower_second = np.array([h_low, s_min, v_min])
        upper_second = np.array([179, s_max, v_max])
        first_mask = cv2.inRange(hsv_image, lower_first, upper_first)
        second_mask = cv2.inRange(hsv_image, lower_second, upper_second)
        return cv2.bitwise_or(first_mask, second_mask)

    def clean_mask(self, mask):
        kernel_size = max(0, int(self.tuning['morphology_kernel_size']))
        if kernel_size <= 1:
            return mask.copy()

        if kernel_size % 2 == 0:
            kernel_size += 1

        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    def find_tape_contours(self, cleaned_mask):
        contours, _hierarchy = cv2.findContours(
            cleaned_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        min_area = max(0, int(self.tuning['minimum_contour_area']))
        return [
            contour for contour in contours
            if cv2.contourArea(contour) >= min_area
        ]

    def build_approach_roi_mask(self, cleaned_mask):
        if not self.approach_roi_enabled:
            return cleaned_mask.copy()

        roi_mask = np.zeros_like(cleaned_mask)
        x_min, y_min, x_max, y_max = self.approach_roi_bounds(
            cleaned_mask.shape
        )
        if x_max <= x_min or y_max <= y_min:
            return roi_mask

        roi_mask[y_min:y_max, x_min:x_max] = cleaned_mask[
            y_min:y_max,
            x_min:x_max,
        ]
        return roi_mask

    def approach_roi_bounds(self, mask_shape):
        height, width = mask_shape[:2]
        x_min = min(self.approach_roi_x_min, self.approach_roi_x_max)
        x_max = max(self.approach_roi_x_min, self.approach_roi_x_max)
        y_min = min(self.approach_roi_y_min, self.approach_roi_y_max)
        y_max = max(self.approach_roi_y_min, self.approach_roi_y_max)

        return (
            self._clamp(x_min, 0, width),
            self._clamp(y_min, 0, height),
            self._clamp(x_max, 0, width),
            self._clamp(y_max, 0, height),
        )

    def count_blue_components(self, cleaned_mask):
        component_mask = self.build_approach_roi_mask(cleaned_mask)
        label_count, _labels, stats, centroids = (
            cv2.connectedComponentsWithStats(component_mask, connectivity=8)
        )
        min_area = max(0, int(self.approach_min_component_area))
        max_area = max(min_area, int(self.approach_max_component_area))

        components = []
        for label in range(1, label_count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < min_area or area > max_area:
                continue

            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            center_x, center_y = centroids[label]
            if not np.isfinite(center_x) or not np.isfinite(center_y):
                continue

            components.append({
                'area': area,
                'bbox': (x, y, width, height),
                'center': (int(round(center_x)), int(round(center_y))),
            })

        components.sort(key=lambda component: component['area'], reverse=True)
        return len(components), components

    def update_approach_trigger(self, component_count):
        if not self.approach_trigger_enabled:
            self.approach_stable_frame_count = 0
            self.current_approach_state = 'APPROACH'
            return

        target_count = max(0, int(self.approach_target_component_count))
        count_tolerance = max(0, int(self.approach_count_tolerance))
        required_frames = max(
            1,
            int(self.approach_required_stable_frames),
        )
        within_tolerance = (
            abs(int(component_count) - target_count) <= count_tolerance
        )

        if within_tolerance:
            self.approach_stable_frame_count = min(
                self.approach_stable_frame_count + 1,
                required_frames,
            )
        else:
            self.approach_stable_frame_count = 0

        if self.approach_stable_frame_count >= required_frames:
            self.current_approach_state = 'READY_TO_PARK'
        else:
            self.current_approach_state = 'APPROACH'

    def detect_edges(self, cleaned_mask):
        canny_low = self._clamp(self.tuning['canny_low'], 0, 255)
        canny_high = self._clamp(self.tuning['canny_high'], 0, 255)
        low_threshold, high_threshold = sorted((canny_low, canny_high))
        return cv2.Canny(cleaned_mask, low_threshold, high_threshold)

    def detect_line_segments(self, cleaned_mask):
        edges = self.detect_edges(cleaned_mask)
        threshold = max(1, int(self.tuning['hough_threshold']))
        min_line_length = max(0, int(self.tuning['min_line_length']))
        max_line_gap = max(0, int(self.tuning['max_line_gap']))

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=threshold,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap,
        )

        if lines is None:
            return []

        return [
            (int(x1), int(y1), int(x2), int(y2))
            for [[x1, y1, x2, y2]] in lines
        ]

    def build_debug_overlay(self, bgr_image, contours, line_segments):
        overlay = bgr_image.copy()
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
        self.draw_bev_source_quadrilateral(overlay)

        for index, contour in enumerate(contours):
            x, y, width, height = cv2.boundingRect(contour)
            area = int(cv2.contourArea(contour))
            cv2.rectangle(
                overlay,
                (x, y),
                (x + width, y + height),
                (0, 255, 255),
                2,
            )
            cv2.putText(
                overlay,
                f'{index}: {area}',
                (x, max(0, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        for x1, y1, x2, y2 in line_segments:
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(overlay, (x1, y1), 3, (0, 255, 255), -1)
            cv2.circle(overlay, (x2, y2), 3, (0, 255, 255), -1)

        cv2.putText(
            overlay,
            f'line segments: {len(line_segments)}',
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        self.draw_approach_debug(overlay)
        self.draw_motor_calibration_debug(overlay)

        return overlay

    def draw_approach_debug(self, overlay):
        if self.approach_roi_enabled:
            x_min, y_min, x_max, y_max = self.approach_roi_bounds(
                overlay.shape
            )
            if x_max > x_min and y_max > y_min:
                cv2.rectangle(
                    overlay,
                    (x_min, y_min),
                    (max(x_min, x_max - 1), max(y_min, y_max - 1)),
                    (0, 165, 255),
                    2,
                )

        for index, component in enumerate(self.current_approach_components):
            x, y, width, height = component['bbox']
            center_x, center_y = component['center']
            cv2.rectangle(
                overlay,
                (x, y),
                (x + width, y + height),
                (255, 128, 0),
                2,
            )
            cv2.circle(overlay, (center_x, center_y), 4, (0, 0, 255), -1)
            cv2.putText(
                overlay,
                f'C{index}',
                (x, max(0, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 128, 0),
                1,
                cv2.LINE_AA,
            )

        required_frames = max(1, int(self.approach_required_stable_frames))
        state_color = (0, 255, 0)
        if self.current_approach_state != 'READY_TO_PARK':
            state_color = (0, 255, 255)

        text_rows = [
            (
                f'components: {self.current_approach_component_count}',
                (255, 128, 0),
            ),
            (
                f'approach state: {self.current_approach_state}',
                state_color,
            ),
            (
                'stable frames: '
                f'{self.approach_stable_frame_count} / {required_frames}',
                state_color,
            ),
            (
                'approach trigger: '
                f'{"ON" if self.approach_trigger_enabled else "OFF"}',
                state_color,
            ),
        ]

        y_position = 60
        for text, color in text_rows:
            cv2.putText(
                overlay,
                text,
                (20, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )
            y_position += 28

    def draw_motor_calibration_debug(
        self,
        overlay,
        x_position=20,
        y_position=190,
    ):
        active_pulse = self.active_motor_pulse
        if active_pulse is None:
            active_pulse = 'none'

        allowed = self.motor_motion_allowed()
        mode_text = 'ON' if self.motor_calibration_enabled else 'OFF'
        motion_text = 'true' if self.motion_enabled else 'false'
        preview = self.last_motor_command_preview
        text_color = (0, 255, 0) if allowed else (0, 255, 255)
        remaining_time = self.motor_pulse_remaining_sec()
        warning_text = ''
        if not self.motion_enabled:
            warning_text = 'SAFETY: motion disabled - preview only'
        elif not self.motor_calibration_enabled:
            warning_text = 'SAFETY: calibration mode off - zero cmd_vel'

        odom_mode_text = 'ON' if self.odom_control_enabled else 'OFF'
        rc_mode_text = 'ON' if self.rc_control_enabled else 'OFF'
        odom_ready_text = 'true' if self.odom_is_recent() else 'false'
        text_rows = [
            f'motor calibration mode: {mode_text}',
            f'motion enabled: {motion_text}',
            f'rc control: {rc_mode_text}',
            f'rc command: linear.x={self.rc_linear_x:.3f}, '
            f'angular.z={self.rc_angular_z:.3f}',
            'rc keys: f/b throttle, j/k steer, c center, v throttle0, '
            'z stop, x e-stop',
            f'odom control: {odom_mode_text}',
            f'odom ready: {odom_ready_text} frames={self.odom_frame_count}',
            f'vesc state ready: {self.vesc_state_is_recent()} '
            f'frames={self.vesc_state_frame_count}',
            f'feedback source: {self.odom_motion_feedback_source}',
            f'vesc speed: {self.latest_vesc_speed_mps:.3f} m/s',
            f'active odom motion: {self.active_odom_motion or "none"}',
            f'odom distance: {self.current_odom_distance_m():.3f} / '
            f'{self.odom_motion_target_distance_m:.3f}m',
            f'odom stop at: {self.odom_motion_completion_distance_m():.3f}m',
            f'odom speed profile: {self.odom_speed_profile_enabled}',
            f'cmd_vel subscribers: {self.count_subscribers(self.cmd_vel_topic)}',
            f'ackermann subscribers: '
            f'{self.count_subscribers(self.ackermann_topic)}',
            f'active sequence: {self.active_motor_sequence or "none"}',
            f'sequence step: {self.motor_sequence_step_text()}',
            f'active pulse: {active_pulse}',
            'last command preview: '
            f'linear.x={preview["linear_x"]:.3f}, '
            f'angular.z={preview["angular_z"]:.3f}',
            f'ackermann topic: {self.ackermann_topic}',
            'motor settings: '
            f'linear={self.test_linear_speed:.3f}, '
            f'backward={self.test_backward_speed:.3f}, '
            f'left={self.test_angular_left:.3f}, '
            f'right={self.test_angular_right:.3f}',
            f'pulse duration: {self.test_pulse_duration_sec:.2f}s',
            f'pulse remaining: {remaining_time:.2f}s',
            f'last motor status: {self.last_motor_command_status}',
        ]

        for text in text_rows:
            cv2.putText(
                overlay,
                text,
                (x_position, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                text_color,
                2,
                cv2.LINE_AA,
            )
            y_position += 25

        if warning_text:
            cv2.putText(
                overlay,
                warning_text,
                (x_position, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

    def draw_bev_source_quadrilateral(self, overlay):
        if not self.bev_enabled:
            return

        points = self.bev_source_points(overlay.shape)
        cv2.polylines(overlay, [points.astype(np.int32)], True, (255, 0, 255), 2)

        for index, point in enumerate(points.astype(np.int32), start=1):
            x, y = point
            cv2.circle(overlay, (x, y), 5, (255, 0, 255), -1)
            cv2.putText(
                overlay,
                f'p{index}',
                (x + 6, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 255),
                1,
                cv2.LINE_AA,
            )

    def compute_bird_eye_view(self, frame, cleaned_mask):
        bev_width = max(1, int(self.bev_width))
        bev_height = max(1, int(self.bev_height))

        if not self.bev_enabled:
            empty_image = np.zeros((bev_height, bev_width, 3), dtype=np.uint8)
            empty_mask = np.zeros((bev_height, bev_width), dtype=np.uint8)
            return empty_image, empty_mask

        source_points = self.bev_source_points(frame.shape)
        destination_points = np.float32([
            [0, 0],
            [bev_width, 0],
            [bev_width, bev_height],
            [0, bev_height],
        ])

        transform = cv2.getPerspectiveTransform(
            source_points,
            destination_points,
        )
        bev_image = cv2.warpPerspective(
            frame,
            transform,
            (bev_width, bev_height),
        )
        bev_mask = cv2.warpPerspective(
            cleaned_mask,
            transform,
            (bev_width, bev_height),
        )
        return bev_image, bev_mask

    def bev_source_points(self, frame_shape):
        height, width = frame_shape[:2]
        points = np.float32([
            [self.tuning['bev_src_p1_x'], self.tuning['bev_src_p1_y']],
            [self.tuning['bev_src_p2_x'], self.tuning['bev_src_p2_y']],
            [self.tuning['bev_src_p3_x'], self.tuning['bev_src_p3_y']],
            [self.tuning['bev_src_p4_x'], self.tuning['bev_src_p4_y']],
        ])
        points[:, 0] = np.clip(points[:, 0], 0, max(0, width - 1))
        points[:, 1] = np.clip(points[:, 1], 0, max(0, height - 1))
        return points.astype(np.float32)

    def build_bev_line_overlay(self, bev_image, bev_line_segments):
        overlay = bev_image.copy()
        for x1, y1, x2, y2 in bev_line_segments:
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(overlay, (x1, y1), 3, (0, 255, 255), -1)
            cv2.circle(overlay, (x2, y2), 3, (0, 255, 255), -1)

        cv2.putText(
            overlay,
            f'BEV line segments: {len(bev_line_segments)}',
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return overlay

    def update_extension_hooks(
        self,
        _bgr_image,
        _cleaned_mask,
        _contours,
        _line_segments,
        bev_image,
        _bev_mask,
        _bev_line_segments,
    ):
        slot_polygons = self.generate_parking_slot_polygons(bev_image)
        occupancy = self.classify_slot_occupancy(bev_image, slot_polygons)
        target_slot = self.select_target_slot(occupancy)
        self.plan_parking_waypoints(target_slot)
        return None

    def generate_parking_slot_polygons(self, _bev_image):
        return []

    def classify_slot_occupancy(self, _bev_image, _slot_polygons):
        return []

    def select_target_slot(self, _occupancy):
        return None

    def plan_parking_waypoints(self, _target_slot):
        return []

    def publish_waypoint_control(self, _waypoints):
        return None

    def _show_debug_windows(
        self,
        bgr_image,
        mask,
        cleaned_mask,
        edges,
        overlay,
        bev_image,
        bev_mask,
        bev_line_overlay,
        approach_roi_mask,
    ):
        if not self.gui_available:
            return

        try:
            if self.motor_calibration_gui_only:
                cv2.imshow(DEBUG_OVERLAY_WINDOW, overlay)
                return

            cv2.imshow(ORIGINAL_WINDOW, bgr_image)
            cv2.imshow(BLUE_MASK_WINDOW, mask)
            cv2.imshow(CLEANED_MASK_WINDOW, cleaned_mask)
            cv2.imshow(CANNY_EDGES_WINDOW, edges)
            cv2.imshow(DEBUG_OVERLAY_WINDOW, overlay)
            cv2.imshow(BEV_IMAGE_WINDOW, bev_image)
            cv2.imshow(BEV_MASK_WINDOW, bev_mask)
            cv2.imshow(BEV_LINE_OVERLAY_WINDOW, bev_line_overlay)
            cv2.imshow(APPROACH_ROI_MASK_WINDOW, approach_roi_mask)
        except cv2.error as error:
            self.gui_available = False
            self.get_logger().warning(
                'OpenCV GUI failed while showing frames. '
                f'Continuing headless: {error}'
            )

    def _show_status_windows_if_needed(self):
        if self.latest_original is not None:
            return

        status_image = self._build_status_image()
        empty_mask = np.zeros(status_image.shape[:2], dtype=np.uint8)
        if self.motor_calibration_gui_only:
            cv2.imshow(DEBUG_OVERLAY_WINDOW, status_image)
            return

        cv2.imshow(ORIGINAL_WINDOW, status_image)
        cv2.imshow(BLUE_MASK_WINDOW, empty_mask)
        cv2.imshow(CLEANED_MASK_WINDOW, empty_mask)
        cv2.imshow(CANNY_EDGES_WINDOW, empty_mask)
        cv2.imshow(DEBUG_OVERLAY_WINDOW, status_image)
        cv2.imshow(BEV_IMAGE_WINDOW, status_image)
        cv2.imshow(BEV_MASK_WINDOW, empty_mask)
        cv2.imshow(BEV_LINE_OVERLAY_WINDOW, status_image)
        cv2.imshow(APPROACH_ROI_MASK_WINDOW, empty_mask)

    def _build_status_image(self):
        image = np.zeros((620, 900, 3), dtype=np.uint8)
        y_position = 40

        for line in self._camera_status_lines():
            cv2.putText(
                image,
                line,
                (20, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            y_position += 28

        y_position += 16
        cv2.putText(
            image,
            'Motor calibration can be tested without camera frames.',
            (20, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        self.draw_motor_calibration_debug(
            image,
            x_position=20,
            y_position=y_position + 34,
        )

        return image

    def _camera_status_lines(self):
        if not self.perception_enabled:
            return [
                'Perception disabled for motor/odom control.',
                'Camera subscription, OpenCV fallback, HSV, Canny, Hough,',
                'BEV, and approach-trigger processing are not running.',
            ]

        publisher_count = self.count_publishers(self.camera_topic)
        lines = [
            'No camera frame received yet.',
            f'ROS topic: {self.camera_topic}',
            f'ROS publishers on topic: {publisher_count}',
            f'ROS frames received: {self.ros_frame_count}',
            f'OpenCV frames received: {self.opencv_frame_count}',
            f'OpenCV fallback enabled: {self.use_opencv_camera_fallback}',
            f'OpenCV sources tried: {self._opencv_sources_to_try()}',
        ]

        if publisher_count == 0:
            lines.append('No ROS camera publisher is visible on this topic.')
        else:
            lines.append('Publisher is visible, but no Image messages arrived.')
            lines.append('Check topic type, QoS, or camera publisher health.')

        lines.append('If using direct webcam, check: ls -l /dev/video*')
        return lines

    def _handle_keyboard(self, key):
        if key == 255:
            return
        if key == ord('q'):
            self.request_shutdown()
        elif key == ord('s'):
            self.pending_autosave = False
            self.save_parameters()
        elif key == ord('l'):
            self.load_parameters()
        elif key == 32:
            self.save_debug_images()
        elif key == ord('a'):
            self.toggle_approach_trigger()
        elif key == ord('p'):
            self.print_approach_status()
        elif key == ord('m'):
            self.toggle_motor_calibration()
        elif key == ord('e'):
            self.toggle_motion_enabled()
        elif self.rc_control_enabled and self.handle_rc_keyboard(key):
            return
        elif key == ord('f'):
            self.cancel_motor_sequence('manual forward override')
            self.cancel_odom_motion('manual forward override')
            if self.odom_control_enabled:
                self.start_odom_motion(
                    'odom forward',
                    self.test_linear_speed,
                    self.straight_correction_angular_z,
                    self.odom_forward_distance_m,
                )
            else:
                self.start_motor_pulse(
                    'forward',
                    self.test_linear_speed,
                    self.straight_correction_angular_z,
                )
        elif key == ord('b'):
            self.cancel_motor_sequence('manual backward override')
            self.cancel_odom_motion('manual backward override')
            if self.odom_control_enabled:
                self.start_odom_motion(
                    'odom backward',
                    -abs(self.test_backward_speed),
                    self.straight_correction_angular_z,
                    self.odom_backward_distance_m,
                )
            else:
                self.start_motor_pulse(
                    'backward',
                    -abs(self.test_backward_speed),
                    self.straight_correction_angular_z,
                )
        elif key == ord('j'):
            self.cancel_motor_sequence('manual right override')
            self.cancel_odom_motion('manual right override')
            if self.odom_control_enabled:
                self.start_odom_motion(
                    'odom right',
                    self.test_linear_speed,
                    self.test_angular_right,
                    self.odom_turn_distance_m,
                )
            else:
                self.start_motor_pulse(
                    'right',
                    self.test_linear_speed,
                    self.test_angular_right,
                )
        elif key == ord('k'):
            self.cancel_motor_sequence('manual left override')
            self.cancel_odom_motion('manual left override')
            if self.odom_control_enabled:
                self.start_odom_motion(
                    'odom left',
                    self.test_linear_speed,
                    self.test_angular_left,
                    self.odom_turn_distance_m,
                )
            else:
                self.start_motor_pulse(
                    'left',
                    self.test_linear_speed,
                    self.test_angular_left,
                )
        elif key == ord('1'):
            self.start_hardcoded_parking_sequence('sequence_1')
        elif key == ord('2'):
            self.start_hardcoded_parking_sequence('sequence_2')
        elif key == ord('3'):
            self.start_hardcoded_parking_sequence('sequence_1_reverse')
        elif key == ord('4'):
            self.start_hardcoded_parking_sequence('sequence_2_reverse')
        elif key == ord('x'):
            self.emergency_stop()

    def keyboard_topic_callback(self, msg):
        if not msg.data:
            return

        self._handle_keyboard(ord(msg.data[0]))

    def handle_rc_keyboard(self, key):
        handled = True
        if key == ord('f'):
            self.rc_linear_x += abs(float(self.rc_linear_step))
        elif key == ord('b'):
            self.rc_linear_x -= abs(float(self.rc_linear_step))
        elif key == ord('j'):
            self.rc_angular_z -= abs(float(self.rc_angular_step))
        elif key == ord('k'):
            self.rc_angular_z += abs(float(self.rc_angular_step))
        elif key == ord('c'):
            self.rc_angular_z = 0.0
        elif key == ord('v'):
            self.rc_linear_x = 0.0
        elif key == ord('z'):
            self.rc_linear_x = 0.0
            self.rc_angular_z = 0.0
        else:
            handled = False

        if not handled:
            return False

        self.cancel_motor_sequence('rc keyboard override')
        self.cancel_odom_motion('rc keyboard override')
        self.clear_motor_pulse('rc keyboard override')
        self.rc_linear_x = self._clamp_float(
            self.rc_linear_x,
            -abs(float(self.rc_max_linear_speed)),
            abs(float(self.rc_max_linear_speed)),
        )
        self.rc_angular_z = self._clamp_float(
            self.rc_angular_z,
            -abs(float(self.rc_max_angular_z)),
            abs(float(self.rc_max_angular_z)),
        )
        self.last_motor_command_preview = {
            'linear_x': self.rc_linear_x,
            'angular_z': self.rc_angular_z,
        }

        if self.motor_motion_allowed():
            self.last_motor_command_status = 'rc active'
            self.publish_motor_command(self.rc_linear_x, self.rc_angular_z)
        else:
            self.last_motor_command_status = 'rc preview blocked'
            self.publish_zero_velocity()

        self.get_logger().warning(
            f'RC command: linear.x={self.rc_linear_x:.3f}, '
            f'angular.z={self.rc_angular_z:.3f}, '
            f'allowed={self.motor_motion_allowed()}'
        )
        return True

    def toggle_approach_trigger(self):
        self.approach_trigger_enabled = not self.approach_trigger_enabled
        if not self.approach_trigger_enabled:
            self.approach_stable_frame_count = 0
            self.current_approach_state = 'APPROACH'

        self.get_logger().info(
            'approach_trigger_enabled: '
            f'{self.approach_trigger_enabled}'
        )

    def print_approach_status(self):
        lines = [
            f'approach_trigger_enabled: {self.approach_trigger_enabled}',
            f'approach_roi_enabled: {self.approach_roi_enabled}',
            f'component count: {self.current_approach_component_count}',
            f'approach state: {self.current_approach_state}',
            'stable frames: '
            f'{self.approach_stable_frame_count} / '
            f'{max(1, int(self.approach_required_stable_frames))}',
        ]

        for index, component in enumerate(self.current_approach_components):
            lines.append(
                f'component {index}: '
                f'area={component["area"]}, '
                f'bbox={component["bbox"]}, '
                f'center={component["center"]}'
            )

        self.get_logger().info('\n' + '\n'.join(lines))

    def toggle_motor_calibration(self):
        self.motor_calibration_enabled = not self.motor_calibration_enabled
        if not self.motor_calibration_enabled:
            self.motion_enabled = False
            self.cancel_odom_motion('calibration off')
            self.clear_motor_pulse('calibration off')
            self.rc_linear_x = 0.0
            self.rc_angular_z = 0.0
            self.publish_zero_velocity()

        self.get_logger().warning(
            'motor_calibration_enabled: '
            f'{self.motor_calibration_enabled}. '
            f'motion_enabled: {self.motion_enabled}'
        )

    def toggle_motion_enabled(self):
        self.motion_enabled = not self.motion_enabled
        if not self.motion_enabled:
            self.cancel_odom_motion('motion disabled')
            self.clear_motor_pulse('motion disabled')
            self.rc_linear_x = 0.0
            self.rc_angular_z = 0.0
            self.publish_zero_velocity()

        if self.motion_enabled and not self.motor_calibration_enabled:
            self.get_logger().warning(
                'motion_enabled is true, but nonzero /cmd_vel is still '
                'blocked because motor_calibration_enabled is false.'
            )
            return

        self.get_logger().warning(f'motion_enabled: {self.motion_enabled}')

    def _read_trackbars(self):
        if self.motor_calibration_gui_only:
            return False

        changed = False
        for label, parameter_name, max_value in TRACKBAR_SPECS:
            value = cv2.getTrackbarPos(label, TRACKBAR_WINDOW)
            value = self._clamp(value, 0, max_value)
            if value != self.tuning[parameter_name]:
                self.tuning[parameter_name] = value
                changed = True

        if changed and self.auto_save_on_trackbar_change:
            self.pending_autosave = True
            self.last_trackbar_change_time = time.monotonic()

        return changed

    def _read_motor_trackbars(self):
        changed = False
        for label, parameter_name, max_value in MOTOR_TRACKBAR_SPECS:
            raw_value = cv2.getTrackbarPos(label, MOTOR_TRACKBAR_WINDOW)
            raw_value = self._clamp(raw_value, 0, max_value)
            value = self._motor_trackbar_to_value(
                parameter_name,
                raw_value,
            )
            if abs(value - getattr(self, parameter_name)) > 1e-9:
                setattr(self, parameter_name, value)
                changed = True

        if changed and self.auto_save_on_trackbar_change:
            self.pending_autosave = True
            self.last_trackbar_change_time = time.monotonic()

        return changed

    def _motor_trackbar_to_value(self, parameter_name, raw_value):
        if parameter_name == 'test_pulse_duration_sec':
            return float(raw_value) / 1000.0

        value = float(raw_value) / 1000.0
        if parameter_name == 'test_angular_right':
            return -value
        return value

    def _motor_value_to_trackbar(self, parameter_name, max_value):
        value = getattr(self, parameter_name)
        if parameter_name == 'test_pulse_duration_sec':
            raw_value = int(round(float(value) * 1000.0))
        elif parameter_name == 'test_angular_right':
            raw_value = int(round(abs(float(value)) * 1000.0))
        else:
            raw_value = int(round(float(value) * 1000.0))

        return self._clamp(raw_value, 0, max_value)

    def _maybe_auto_save(self):
        if not self.pending_autosave:
            return

        elapsed = time.monotonic() - self.last_trackbar_change_time
        if elapsed < self.auto_save_debounce_sec:
            return

        self.pending_autosave = False
        self.save_parameters(auto_save=True)

    def safety_timer_callback(self):
        if self.rc_control_enabled:
            self.update_rc_control()
            return

        if self.active_odom_motion is not None:
            self.update_odom_motion()
            return

        if self.active_motor_pulse is None:
            self.publish_zero_velocity()
            return

        if not self.motor_motion_allowed():
            self.cancel_motor_sequence('blocked during sequence')
            self.clear_motor_pulse('blocked during pulse')
            self.publish_zero_velocity()
            return

        if self.motor_pulse_start_time is None:
            self.cancel_motor_sequence('invalid sequence pulse')
            self.clear_motor_pulse('invalid pulse')
            self.publish_zero_velocity()
            return

        pulse_duration = self.current_motor_pulse_duration_sec()
        elapsed = time.monotonic() - self.motor_pulse_start_time
        if elapsed <= pulse_duration:
            self.publish_motor_command(
                self.last_motor_command_preview['linear_x'],
                self.last_motor_command_preview['angular_z'],
            )
            return

        self.complete_motor_pulse()

    def motor_motion_allowed(self):
        return self.motor_calibration_enabled and self.motion_enabled

    def update_rc_control(self):
        if not self.motor_motion_allowed():
            self.publish_zero_velocity()
            return

        self.last_motor_command_preview = {
            'linear_x': self.rc_linear_x,
            'angular_z': self.rc_angular_z,
        }
        self.last_motor_command_status = 'rc active'
        self.publish_motor_command(self.rc_linear_x, self.rc_angular_z)

    def current_motor_pulse_duration_sec(self):
        if self.active_motor_pulse_duration_sec is None:
            return max(0.0, float(self.test_pulse_duration_sec))

        return max(0.0, float(self.active_motor_pulse_duration_sec))

    def motor_pulse_remaining_sec(self):
        if self.active_motor_pulse is None or self.motor_pulse_start_time is None:
            return 0.0

        pulse_duration = self.current_motor_pulse_duration_sec()
        elapsed = time.monotonic() - self.motor_pulse_start_time
        return max(0.0, pulse_duration - elapsed)

    def odom_callback(self, msg):
        pose = msg.pose.pose
        orientation = pose.orientation
        yaw = self.yaw_from_quaternion(
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        )
        self.latest_odom_pose = {
            'x': float(pose.position.x),
            'y': float(pose.position.y),
            'yaw': yaw,
        }
        self.latest_odom_time = time.monotonic()
        self.odom_frame_count += 1
        if self.active_odom_motion is not None:
            self.update_odom_motion()

    def vesc_state_callback(self, msg):
        now = time.monotonic()
        raw_speed = float(msg.state.speed)
        if abs(self.vesc_speed_to_erpm_gain) < 1e-9:
            speed_mps = 0.0
        else:
            speed_mps = (
                (raw_speed - self.vesc_speed_to_erpm_offset) /
                self.vesc_speed_to_erpm_gain
            )
        if abs(speed_mps) < 0.05:
            speed_mps = 0.0

        if self.latest_vesc_state_monotonic is not None:
            dt = max(0.0, now - self.latest_vesc_state_monotonic)
            self.integrated_vesc_distance_m += abs(speed_mps) * dt

        self.latest_vesc_speed_mps = speed_mps
        self.latest_vesc_state_monotonic = now
        self.latest_vesc_state_time = now
        self.vesc_state_frame_count += 1
        if (
            self.active_odom_motion is not None and
            self.odom_motion_feedback_source == 'vesc_state'
        ):
            self.update_odom_motion()

    def odom_is_recent(self):
        if self.latest_odom_pose is None or self.latest_odom_time is None:
            return False

        elapsed = time.monotonic() - self.latest_odom_time
        return elapsed <= max(0.0, float(self.odom_timeout_sec))

    def vesc_state_is_recent(self):
        if self.latest_vesc_state_time is None:
            return False

        elapsed = time.monotonic() - self.latest_vesc_state_time
        return elapsed <= max(0.0, float(self.vesc_state_timeout_sec))

    def odom_motion_feedback_available(self):
        if self.odom_is_recent():
            return True

        return (
            self.use_vesc_state_distance_fallback and
            self.vesc_state_is_recent()
        )

    def choose_odom_motion_feedback_source(self):
        if self.odom_is_recent():
            return 'odom'
        if (
            self.use_vesc_state_distance_fallback and
            self.vesc_state_is_recent()
        ):
            return 'vesc_state'
        return 'none'

    def current_odom_distance_m(self):
        if self.odom_motion_feedback_source == 'vesc_state':
            return max(
                0.0,
                self.integrated_vesc_distance_m -
                self.odom_motion_start_vesc_distance_m,
            )

        if self.latest_odom_pose is None or self.odom_motion_start_pose is None:
            return 0.0

        dx = self.latest_odom_pose['x'] - self.odom_motion_start_pose['x']
        dy = self.latest_odom_pose['y'] - self.odom_motion_start_pose['y']
        return math.hypot(dx, dy)

    def odom_motion_remaining_m(self):
        if self.active_odom_motion is None:
            return 0.0

        return max(
            0.0,
            self.odom_motion_target_distance_m -
            self.current_odom_distance_m(),
        )

    def odom_motion_completion_distance_m(self):
        if self.active_odom_motion is None:
            return 0.0

        stop_buffer = max(0.0, float(self.odom_stop_buffer_m))
        target = max(0.0, float(self.odom_motion_target_distance_m))
        return max(0.0, target - min(stop_buffer, target))

    def active_odom_motion_command(self):
        linear_x = float(self.last_motor_command_preview['linear_x'])
        angular_z = float(self.last_motor_command_preview['angular_z'])

        if not self.odom_speed_profile_enabled:
            return linear_x, angular_z

        completion_distance = self.odom_motion_completion_distance_m()
        distance = self.current_odom_distance_m()
        remaining = max(0.0, completion_distance - distance)
        slowdown_distance = max(1e-6, float(self.odom_slowdown_distance_m))
        min_scale = self._clamp_float(
            float(self.odom_min_speed_scale),
            0.0,
            1.0,
        )
        scale = self._clamp_float(
            remaining / slowdown_distance,
            min_scale,
            1.0,
        )

        if abs(linear_x) < 1e-9:
            return linear_x, angular_z

        command_linear = linear_x * scale
        min_linear = max(0.0, float(self.odom_min_linear_speed))
        max_linear = abs(linear_x)
        command_abs = min(max_linear, max(abs(command_linear), min_linear))
        command_linear = math.copysign(command_abs, linear_x)
        return command_linear, angular_z

    def start_odom_motion(
        self,
        motion_name,
        linear_x,
        angular_z,
        target_distance_m,
        timeout_sec=None,
    ):
        self.last_motor_command_preview = {
            'linear_x': float(linear_x),
            'angular_z': float(angular_z),
        }
        target_distance_m = abs(float(target_distance_m))
        if timeout_sec is None:
            timeout_sec = self.odom_segment_timeout_sec
        timeout_sec = max(0.0, float(timeout_sec))

        allowed = self.motor_motion_allowed()
        feedback_source = self.choose_odom_motion_feedback_source()
        odom_ready = feedback_source != 'none'
        cmd_vel_subscribers = self.count_subscribers(self.cmd_vel_topic)
        ackermann_subscribers = 0
        if self.publish_ackermann:
            ackermann_subscribers = self.count_subscribers(
                self.ackermann_topic
            )
        ackermann_cmd_subscribers = self.count_subscribers('/ackermann_cmd')
        ackermann_cmd_out_subscribers = self.count_subscribers(
            '/ackermann_cmd_out'
        )
        motor_speed_subscribers = self.count_subscribers(
            '/commands/motor/speed'
        )
        motor_unsmoothed_speed_subscribers = self.count_subscribers(
            '/commands/motor/unsmoothed_speed'
        )
        servo_position_subscribers = self.count_subscribers(
            '/commands/servo/position'
        )
        servo_unsmoothed_position_subscribers = self.count_subscribers(
            '/commands/servo/unsmoothed_position'
        )
        odom_publishers = self.count_publishers(self.odom_topic)
        vesc_state_publishers = self.count_publishers(self.vesc_state_topic)

        self.get_logger().warning(
            f'Odom motion key received: {motion_name}. '
            f'linear.x={linear_x:.3f}, angular.z={angular_z:.3f}, '
            f'target={target_distance_m:.3f}m, '
            f'stop_at={max(0.0, target_distance_m - self.odom_stop_buffer_m):.3f}m, '
            f'timeout={timeout_sec:.3f}s, '
            f'allowed={allowed}, odom_ready={odom_ready}, '
            f'feedback_source={feedback_source}, '
            f'cmd_vel_subscribers={cmd_vel_subscribers}, '
            f'ackermann_subscribers={ackermann_subscribers}, '
            f'ackermann_cmd_subscribers={ackermann_cmd_subscribers}, '
            f'ackermann_cmd_out_subscribers='
            f'{ackermann_cmd_out_subscribers}, '
            f'motor_speed_subscribers={motor_speed_subscribers}, '
            f'motor_unsmoothed_speed_subscribers='
            f'{motor_unsmoothed_speed_subscribers}, '
            f'servo_position_subscribers={servo_position_subscribers}, '
            f'servo_unsmoothed_position_subscribers='
            f'{servo_unsmoothed_position_subscribers}, '
            f'odom_publishers={odom_publishers}, '
            f'vesc_state_publishers={vesc_state_publishers}'
        )

        if not allowed:
            self.clear_odom_motion('odom preview blocked')
            self.publish_zero_velocity()
            self.get_logger().warning(
                'Odom motion preview only. Nonzero motion is blocked until '
                'motor_calibration_enabled and motion_enabled are both true.'
            )
            return False

        if not odom_ready:
            self.clear_odom_motion('odom unavailable')
            self.publish_zero_velocity()
            self.get_logger().warning(
                f'Odom motion blocked. No recent odom on {self.odom_topic} '
                f'or VESC state on {self.vesc_state_topic}.'
            )
            return False

        if cmd_vel_subscribers == 0 and ackermann_subscribers == 0:
            self.last_motor_command_status = 'no motor command subscriber'
            self.get_logger().warning(
                f'No subscribers on {self.cmd_vel_topic} or '
                f'{self.ackermann_topic}. Start the actuator bridge launch '
                'before expecting hardware motion.'
            )

        self.clear_motor_pulse('odom motion active')
        self.active_odom_motion = motion_name
        self.odom_motion_feedback_source = feedback_source
        if feedback_source == 'odom':
            self.odom_motion_start_pose = dict(self.latest_odom_pose)
        else:
            self.odom_motion_start_pose = None
        self.odom_motion_start_vesc_distance_m = self.integrated_vesc_distance_m
        self.odom_motion_start_time = time.monotonic()
        self.odom_motion_target_distance_m = target_distance_m
        self.odom_motion_timeout_sec = timeout_sec
        self.last_motor_command_status = f'{motion_name} odom active'
        command_linear, command_angular = self.active_odom_motion_command()
        self.get_logger().warning(
            f'Odom first published command: '
            f'linear.x={command_linear:.3f}, '
            f'angular.z={command_angular:.3f}, '
            f'speed_profile_enabled={self.odom_speed_profile_enabled}, '
            f'slowdown_distance={self.odom_slowdown_distance_m:.3f}m, '
            f'min_speed_scale={self.odom_min_speed_scale:.2f}, '
            f'min_linear_speed={self.odom_min_linear_speed:.3f}'
        )
        self.publish_motor_command(command_linear, command_angular)
        return True

    def update_odom_motion(self):
        if self.active_odom_motion is None:
            return

        if not self.motor_motion_allowed():
            self.cancel_odom_motion('blocked during odom motion')
            self.publish_zero_velocity()
            return

        if (
            self.odom_motion_feedback_source == 'odom' and
            not self.odom_is_recent()
        ):
            self.cancel_odom_motion('odom timeout')
            self.publish_zero_velocity()
            self.get_logger().warning('Odom motion stopped: odom timeout.')
            return

        if (
            self.odom_motion_feedback_source == 'vesc_state' and
            not self.vesc_state_is_recent()
        ):
            self.cancel_odom_motion('vesc state timeout')
            self.publish_zero_velocity()
            self.get_logger().warning(
                'Odom motion stopped: VESC state timeout.'
            )
            return

        elapsed = time.monotonic() - self.odom_motion_start_time
        if elapsed > self.odom_motion_timeout_sec:
            status = (
                f'{self.active_odom_motion} timeout at '
                f'{self.current_odom_distance_m():.3f}m'
            )
            self.cancel_odom_motion(status)
            self.publish_zero_velocity()
            self.get_logger().warning(status)
            return

        distance = self.current_odom_distance_m()
        completion_distance = self.odom_motion_completion_distance_m()
        if distance >= completion_distance:
            completed_motion = self.active_odom_motion
            self.clear_odom_motion(
                f'{completed_motion} complete '
                f'{distance:.3f}m / '
                f'{self.odom_motion_target_distance_m:.3f}m'
            )
            self.publish_zero_velocity()
            self.get_logger().warning(
                f'{completed_motion} complete at {distance:.3f}m.'
            )
            if self.active_motor_sequence is not None:
                self.motor_sequence_step_index += 1
                self.start_current_motor_sequence_step()
            return

        command_linear, command_angular = self.active_odom_motion_command()
        self.publish_motor_command(command_linear, command_angular)

    def clear_odom_motion(self, status=None):
        self.active_odom_motion = None
        self.odom_motion_start_pose = None
        self.odom_motion_start_vesc_distance_m = 0.0
        self.odom_motion_feedback_source = 'none'
        self.odom_motion_start_time = None
        self.odom_motion_target_distance_m = 0.0
        self.odom_motion_timeout_sec = 0.0
        if status is not None:
            self.last_motor_command_status = status

    def cancel_odom_motion(self, status):
        if self.active_odom_motion is None:
            return

        motion_name = self.active_odom_motion
        self.clear_odom_motion(f'{motion_name}: {status}')

    @staticmethod
    def yaw_from_quaternion(x, y, z, w):
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    def start_motor_pulse(
        self,
        pulse_name,
        linear_x,
        angular_z,
        duration_sec=None,
    ):
        self.last_motor_command_preview = {
            'linear_x': float(linear_x),
            'angular_z': float(angular_z),
        }
        if duration_sec is None:
            duration_sec = self.test_pulse_duration_sec
        duration_sec = max(0.0, float(duration_sec))
        allowed = self.motor_motion_allowed()
        cmd_vel_subscribers = self.count_subscribers(self.cmd_vel_topic)
        ackermann_subscribers = 0
        if self.publish_ackermann:
            ackermann_subscribers = self.count_subscribers(
                self.ackermann_topic
            )
        self.get_logger().warning(
            f'Motor pulse key received: {pulse_name}. '
            f'linear.x={linear_x:.3f}, angular.z={angular_z:.3f}, '
            f'duration={duration_sec:.3f}s, '
            f'allowed={allowed}, '
            f'cmd_vel_subscribers={cmd_vel_subscribers}, '
            f'ackermann_subscribers={ackermann_subscribers}'
        )

        if not allowed:
            self.clear_motor_pulse('preview blocked')
            self.publish_zero_velocity()
            self.get_logger().warning(
                'Motor command preview only. Nonzero /cmd_vel is blocked '
                'until motor_calibration_enabled and motion_enabled are both '
                'true.'
            )
            return

        if cmd_vel_subscribers == 0 and ackermann_subscribers == 0:
            self.last_motor_command_status = 'no motor command subscriber'
            self.get_logger().warning(
                f'No subscribers on {self.cmd_vel_topic} or '
                f'{self.ackermann_topic}. Start the actuator bridge launch '
                'before expecting hardware motion.'
            )

        self.active_motor_pulse = pulse_name
        self.motor_pulse_start_time = time.monotonic()
        self.active_motor_pulse_duration_sec = duration_sec
        self.last_motor_command_status = f'{pulse_name} pulse active'
        self.publish_motor_command(
            self.last_motor_command_preview['linear_x'],
            self.last_motor_command_preview['angular_z'],
        )
        self.get_logger().info(
            f'Started {pulse_name} motor pulse: '
            f'linear.x={linear_x:.3f}, angular.z={angular_z:.3f}, '
            f'duration={duration_sec:.3f}s'
        )

    def hardcoded_parking_sequence_steps(self, sequence_name):
        forward_speed = float(self.test_linear_speed)
        backward_speed = -abs(float(self.test_backward_speed))

        if self.odom_control_enabled:
            sequences = {
                'sequence_1': [
                    {
                        'name': (
                            'seq1 forward start '
                            f'{self.sequence_1_forward_start_distance_m:.2f}m'
                        ),
                        'linear_x': forward_speed,
                        'angular_z': self.straight_correction_angular_z,
                        'distance_m':
                            self.sequence_1_forward_start_distance_m,
                    },
                    {
                        'name': (
                            'seq1 j-turn '
                            f'{self.sequence_1_j_turn_1_distance_m:.2f}m'
                        ),
                        'linear_x': forward_speed,
                        'angular_z': self.test_angular_right,
                        'distance_m': self.sequence_1_j_turn_1_distance_m,
                    },
                    {
                        'name': (
                            'seq1 backward '
                            f'{self.sequence_1_backward_distance_m:.2f}m'
                        ),
                        'linear_x': backward_speed,
                        'angular_z': self.straight_correction_angular_z,
                        'distance_m': self.sequence_1_backward_distance_m,
                    },
                    {
                        'name': (
                            'seq1 j-turn '
                            f'{self.sequence_1_j_turn_2_distance_m:.2f}m'
                        ),
                        'linear_x': forward_speed,
                        'angular_z': self.test_angular_right,
                        'distance_m': self.sequence_1_j_turn_2_distance_m,
                    },
                    {
                        'name': (
                            'seq1 backward '
                            f'{self.sequence_1_backward_2_distance_m:.2f}m'
                        ),
                        'linear_x': backward_speed,
                        'angular_z': self.straight_correction_angular_z,
                        'distance_m': self.sequence_1_backward_2_distance_m,
                    },
                    {
                        'name': (
                            'seq1 j-turn '
                            f'{self.sequence_1_j_turn_3_distance_m:.2f}m'
                        ),
                        'linear_x': forward_speed,
                        'angular_z': self.test_angular_right,
                        'distance_m': self.sequence_1_j_turn_3_distance_m,
                    },
                ],
                'sequence_2': [
                    {
                        'name': (
                            'seq2 forward start '
                            f'{self.sequence_2_forward_start_distance_m:.2f}m'
                        ),
                        'linear_x': forward_speed,
                        'angular_z': self.straight_correction_angular_z,
                        'distance_m':
                            self.sequence_2_forward_start_distance_m,
                    },
                    {
                        'name': (
                            'seq2 k-turn '
                            f'{self.sequence_2_k_turn_1_distance_m:.2f}m'
                        ),
                        'linear_x': forward_speed,
                        'angular_z': self.test_angular_left,
                        'distance_m': self.sequence_2_k_turn_1_distance_m,
                    },
                    {
                        'name': (
                            'seq2 backward '
                            f'{self.sequence_2_backward_distance_m:.2f}m'
                        ),
                        'linear_x': backward_speed,
                        'angular_z': self.straight_correction_angular_z,
                        'distance_m': self.sequence_2_backward_distance_m,
                    },
                    {
                        'name': (
                            'seq2 k-turn '
                            f'{self.sequence_2_k_turn_2_distance_m:.2f}m'
                        ),
                        'linear_x': forward_speed,
                        'angular_z': self.test_angular_left,
                        'distance_m': self.sequence_2_k_turn_2_distance_m,
                    },
                    {
                        'name': (
                            'seq2 backward '
                            f'{self.sequence_2_backward_2_distance_m:.2f}m'
                        ),
                        'linear_x': backward_speed,
                        'angular_z': self.straight_correction_angular_z,
                        'distance_m': self.sequence_2_backward_2_distance_m,
                    },
                    {
                        'name': (
                            'seq2 k-turn '
                            f'{self.sequence_2_k_turn_3_distance_m:.2f}m'
                        ),
                        'linear_x': forward_speed,
                        'angular_z': self.test_angular_left,
                        'distance_m': self.sequence_2_k_turn_3_distance_m,
                    },
                ],
            }
            return self._sequence_steps_with_reverse(sequences, sequence_name)

        sequences = {
            'sequence_1': [
                {
                    'name': 'seq1 right 2.0s',
                    'linear_x': forward_speed,
                    'angular_z': self.test_angular_right,
                    'duration_sec': 2.0,
                },
                {
                    'name': 'seq1 backward 0.5s',
                    'linear_x': backward_speed,
                    'angular_z': self.straight_correction_angular_z,
                    'duration_sec': 0.5,
                },
                {
                    'name': 'seq1 right 0.5s',
                    'linear_x': forward_speed,
                    'angular_z': self.test_angular_right,
                    'duration_sec': 0.5,
                },
            ],
            'sequence_2': [
                {
                    'name': 'seq2 left 2.0s',
                    'linear_x': forward_speed,
                    'angular_z': self.test_angular_left,
                    'duration_sec': 2.0,
                },
                {
                    'name': 'seq2 backward 0.5s',
                    'linear_x': backward_speed,
                    'angular_z': self.straight_correction_angular_z,
                    'duration_sec': 0.5,
                },
                {
                    'name': 'seq2 right 0.5s',
                    'linear_x': forward_speed,
                    'angular_z': self.test_angular_right,
                    'duration_sec': 0.5,
                },
            ],
        }
        return self._sequence_steps_with_reverse(sequences, sequence_name)

    def _sequence_steps_with_reverse(self, sequences, sequence_name):
        if sequence_name in sequences:
            return sequences[sequence_name]

        reverse_suffix = '_reverse'
        if not sequence_name.endswith(reverse_suffix):
            return []

        base_sequence_name = sequence_name[:-len(reverse_suffix)]
        base_steps = sequences.get(base_sequence_name)
        if not base_steps:
            return []

        reversed_steps = []
        for step in reversed(base_steps):
            reversed_step = dict(step)
            reversed_step['name'] = f'reverse {step["name"]}'
            reversed_step['linear_x'] = -float(step['linear_x'])
            reversed_steps.append(reversed_step)

        return reversed_steps

    def start_hardcoded_parking_sequence(self, sequence_name):
        self.cancel_odom_motion('hardcoded sequence override')
        if not self.motor_motion_allowed():
            self.cancel_motor_sequence('sequence preview blocked')
            self.publish_zero_velocity()
            self.get_logger().warning(
                f'{sequence_name} blocked. Enable motor calibration with m '
                'and motion with e before running a hardcoded sequence.'
            )
            return

        steps = self.hardcoded_parking_sequence_steps(sequence_name)
        if not steps:
            self.get_logger().warning(f'Unknown motor sequence: {sequence_name}')
            return

        self.active_motor_sequence = sequence_name
        self.motor_sequence_steps = steps
        self.motor_sequence_step_index = 0
        self.get_logger().warning(
            f'Starting {sequence_name}. Press x for emergency stop.'
        )
        self.start_current_motor_sequence_step()

    def start_current_motor_sequence_step(self):
        if self.active_motor_sequence is None:
            return

        if self.motor_sequence_step_index >= len(self.motor_sequence_steps):
            completed_sequence = self.active_motor_sequence
            self.clear_motor_sequence()
            self.clear_motor_pulse(f'{completed_sequence} complete - zero Twist')
            self.publish_zero_velocity()
            self.get_logger().warning(f'{completed_sequence} complete.')
            return

        step = self.motor_sequence_steps[self.motor_sequence_step_index]
        self.get_logger().warning(
            f'{self.active_motor_sequence} step '
            f'{self.motor_sequence_step_index + 1}/'
            f'{len(self.motor_sequence_steps)}: {step["name"]}'
        )

        if self.odom_control_enabled and 'distance_m' in step:
            started = self.start_odom_motion(
                step['name'],
                step['linear_x'],
                step['angular_z'],
                step['distance_m'],
                timeout_sec=step.get('timeout_sec'),
            )
            if not started:
                self.cancel_motor_sequence(
                    f'{step["name"]}: odom step failed'
                )
                self.publish_zero_velocity()
            return

        self.start_motor_pulse(
            step['name'],
            step['linear_x'],
            step['angular_z'],
            duration_sec=step['duration_sec'],
        )

    def complete_motor_pulse(self):
        if self.active_motor_sequence is not None:
            self.motor_sequence_step_index += 1
            self.start_current_motor_sequence_step()
            return

        self.clear_motor_pulse('pulse complete - zero Twist')
        self.publish_zero_velocity()

    def motor_sequence_step_text(self):
        if self.active_motor_sequence is None:
            return 'none'

        total_steps = len(self.motor_sequence_steps)
        if total_steps == 0:
            return 'none'

        current_step = min(self.motor_sequence_step_index + 1, total_steps)
        step_name = self.motor_sequence_steps[
            min(self.motor_sequence_step_index, total_steps - 1)
        ]['name']
        return f'{current_step}/{total_steps} {step_name}'

    def clear_motor_sequence(self):
        self.active_motor_sequence = None
        self.motor_sequence_steps = []
        self.motor_sequence_step_index = 0

    def cancel_motor_sequence(self, status):
        if self.active_motor_sequence is None:
            return

        sequence_name = self.active_motor_sequence
        self.clear_motor_sequence()
        self.last_motor_command_status = f'{sequence_name}: {status}'

    def clear_motor_pulse(self, status=None):
        self.active_motor_pulse = None
        self.motor_pulse_start_time = None
        self.active_motor_pulse_duration_sec = None
        if status is not None:
            self.last_motor_command_status = status

    def emergency_stop(self):
        self.cancel_motor_sequence('emergency stop')
        self.cancel_odom_motion('emergency stop')
        self.clear_motor_pulse('emergency stop')
        self.rc_linear_x = 0.0
        self.rc_angular_z = 0.0
        self.last_motor_command_preview = {
            'linear_x': 0.0,
            'angular_z': 0.0,
        }
        self.publish_zero_velocity()
        self.get_logger().warning('Emergency stop: published zero Twist.')

    def publish_motor_command(self, linear_x, angular_z):
        if not rclpy.ok():
            return

        twist = Twist()
        twist.linear.x = float(linear_x)
        twist.angular.z = float(angular_z)
        try:
            self.cmd_vel_publisher.publish(twist)
        except Exception as error:
            self.get_logger().debug(f'Motor command publish skipped: {error}')

        if not self.publish_ackermann:
            return

        ackermann = AckermannDriveStamped()
        ackermann.header.stamp = self.get_clock().now().to_msg()
        ackermann.drive.speed = float(linear_x)
        ackermann.drive.steering_angle = (
            float(angular_z) * float(self.ackermann_steering_scale)
        )
        try:
            self.ackermann_publisher.publish(ackermann)
        except Exception as error:
            self.get_logger().debug(
                f'Ackermann command publish skipped: {error}'
            )

    def _write_trackbars(self):
        if not self.gui_available:
            return

        if not self.motor_calibration_gui_only:
            for label, parameter_name, max_value in TRACKBAR_SPECS:
                value = self._clamp(self.tuning[parameter_name], 0, max_value)
                cv2.setTrackbarPos(label, TRACKBAR_WINDOW, value)

        for label, parameter_name, max_value in MOTOR_TRACKBAR_SPECS:
            value = self._motor_value_to_trackbar(
                parameter_name,
                max_value,
            )
            cv2.setTrackbarPos(label, MOTOR_TRACKBAR_WINDOW, value)

    def save_parameters(self, auto_save=False):
        parameters = {
            'camera_topic': self.camera_topic,
            'cmd_vel_topic': self.cmd_vel_topic,
            'ackermann_topic': self.ackermann_topic,
            'publish_ackermann': self.publish_ackermann,
            'ackermann_steering_scale': self.ackermann_steering_scale,
            'odom_topic': self.odom_topic,
            'odom_control_enabled': self.odom_control_enabled,
            'odom_timeout_sec': self.odom_timeout_sec,
            'odom_forward_distance_m': self.odom_forward_distance_m,
            'odom_backward_distance_m': self.odom_backward_distance_m,
            'odom_turn_distance_m': self.odom_turn_distance_m,
            'straight_correction_angular_z':
                self.straight_correction_angular_z,
            'sequence_1_forward_start_distance_m':
                self.sequence_1_forward_start_distance_m,
            'sequence_1_j_turn_1_distance_m':
                self.sequence_1_j_turn_1_distance_m,
            'sequence_1_backward_distance_m':
                self.sequence_1_backward_distance_m,
            'sequence_1_j_turn_2_distance_m':
                self.sequence_1_j_turn_2_distance_m,
            'sequence_1_backward_2_distance_m':
                self.sequence_1_backward_2_distance_m,
            'sequence_1_j_turn_3_distance_m':
                self.sequence_1_j_turn_3_distance_m,
            'sequence_2_forward_start_distance_m':
                self.sequence_2_forward_start_distance_m,
            'sequence_2_k_turn_1_distance_m':
                self.sequence_2_k_turn_1_distance_m,
            'sequence_2_backward_distance_m':
                self.sequence_2_backward_distance_m,
            'sequence_2_k_turn_2_distance_m':
                self.sequence_2_k_turn_2_distance_m,
            'sequence_2_backward_2_distance_m':
                self.sequence_2_backward_2_distance_m,
            'sequence_2_k_turn_3_distance_m':
                self.sequence_2_k_turn_3_distance_m,
            'odom_segment_timeout_sec': self.odom_segment_timeout_sec,
            'odom_stop_buffer_m': self.odom_stop_buffer_m,
            'odom_speed_profile_enabled': self.odom_speed_profile_enabled,
            'odom_slowdown_distance_m': self.odom_slowdown_distance_m,
            'odom_min_speed_scale': self.odom_min_speed_scale,
            'odom_min_linear_speed': self.odom_min_linear_speed,
            'use_vesc_state_distance_fallback':
                self.use_vesc_state_distance_fallback,
            'vesc_state_topic': self.vesc_state_topic,
            'vesc_state_timeout_sec': self.vesc_state_timeout_sec,
            'vesc_speed_to_erpm_gain': self.vesc_speed_to_erpm_gain,
            'vesc_speed_to_erpm_offset': self.vesc_speed_to_erpm_offset,
            'safety_timer_period': self.safety_timer_period,
            'perception_enabled': self.perception_enabled,
            'keyboard_topic_control_enabled':
                self.keyboard_topic_control_enabled,
            'keyboard_topic': self.keyboard_topic,
            'enable_gui': self.enable_gui,
            'motor_calibration_gui_only': self.motor_calibration_gui_only,
            'gui_timer_period': self.gui_timer_period,
            'auto_save_on_trackbar_change':
                self.auto_save_on_trackbar_change,
            'auto_save_debounce_sec': self.auto_save_debounce_sec,
            'use_opencv_camera_fallback':
                self.use_opencv_camera_fallback,
            'opencv_camera_source': self.opencv_camera_source,
            'opencv_camera_search_indices':
                self.opencv_camera_search_indices,
            'opencv_camera_index': self.opencv_camera_index,
            'opencv_camera_width': self.opencv_camera_width,
            'opencv_camera_height': self.opencv_camera_height,
            'opencv_timer_period': self.opencv_timer_period,
            'ros_image_timeout': self.ros_image_timeout,
            'debug_image_dir': str(self.debug_image_dir),
            'bev_enabled': self.bev_enabled,
            'bev_width': self.bev_width,
            'bev_height': self.bev_height,
            'approach_trigger_enabled': self.approach_trigger_enabled,
            'approach_roi_enabled': self.approach_roi_enabled,
            'approach_roi_x_min': self.approach_roi_x_min,
            'approach_roi_y_min': self.approach_roi_y_min,
            'approach_roi_x_max': self.approach_roi_x_max,
            'approach_roi_y_max': self.approach_roi_y_max,
            'approach_min_component_area':
                self.approach_min_component_area,
            'approach_max_component_area':
                self.approach_max_component_area,
            'approach_target_component_count':
                self.approach_target_component_count,
            'approach_count_tolerance': self.approach_count_tolerance,
            'approach_required_stable_frames':
                self.approach_required_stable_frames,
            'motor_calibration_enabled':
                self.motor_calibration_enabled,
            'motion_enabled': self.motion_enabled,
            'rc_control_enabled': self.rc_control_enabled,
            'rc_linear_step': self.rc_linear_step,
            'rc_angular_step': self.rc_angular_step,
            'rc_max_linear_speed': self.rc_max_linear_speed,
            'rc_max_angular_z': self.rc_max_angular_z,
            'test_linear_speed': self.test_linear_speed,
            'test_backward_speed': self.test_backward_speed,
            'test_angular_left': self.test_angular_left,
            'test_angular_right': self.test_angular_right,
            'test_pulse_duration_sec': self.test_pulse_duration_sec,
        }
        parameters.update({name: int(self.tuning[name])
                           for name in TUNABLE_PARAMETERS})
        data = {NODE_NAME: {'ros__parameters': parameters}}

        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with self.config_file.open('w', encoding='utf-8') as config:
                yaml.safe_dump(data, config, sort_keys=False)
            action = 'Auto-saved' if auto_save else 'Saved'
            self.get_logger().info(
                f'{action} parameters to {self.config_file}'
            )
        except OSError as error:
            self.get_logger().error(
                f'Could not save parameters to {self.config_file}: {error}'
            )

    def load_parameters(self):
        try:
            with self.config_file.open('r', encoding='utf-8') as config:
                data = yaml.safe_load(config) or {}
        except OSError as error:
            self.get_logger().error(
                f'Could not load parameters from {self.config_file}: {error}'
            )
            return

        node_data = data.get(NODE_NAME, {})
        ros_parameters = node_data.get('ros__parameters', {})
        for name in TUNABLE_PARAMETERS:
            if name in ros_parameters:
                self.tuning[name] = int(ros_parameters[name])

        if 'auto_save_on_trackbar_change' in ros_parameters:
            self.auto_save_on_trackbar_change = bool(
                ros_parameters['auto_save_on_trackbar_change']
            )
        if 'perception_enabled' in ros_parameters:
            self.perception_enabled = bool(
                ros_parameters['perception_enabled']
            )
        if 'keyboard_topic_control_enabled' in ros_parameters:
            self.keyboard_topic_control_enabled = bool(
                ros_parameters['keyboard_topic_control_enabled']
            )
        if 'keyboard_topic' in ros_parameters:
            self.keyboard_topic = str(ros_parameters['keyboard_topic'])
        if 'motor_calibration_gui_only' in ros_parameters:
            self.motor_calibration_gui_only = bool(
                ros_parameters['motor_calibration_gui_only']
            )
        if 'ackermann_topic' in ros_parameters:
            self.ackermann_topic = str(ros_parameters['ackermann_topic'])
        if 'publish_ackermann' in ros_parameters:
            self.publish_ackermann = bool(
                ros_parameters['publish_ackermann']
            )
        if 'ackermann_steering_scale' in ros_parameters:
            self.ackermann_steering_scale = float(
                ros_parameters['ackermann_steering_scale']
            )
        if 'odom_topic' in ros_parameters:
            self.odom_topic = str(ros_parameters['odom_topic'])
        if 'odom_control_enabled' in ros_parameters:
            self.odom_control_enabled = bool(
                ros_parameters['odom_control_enabled']
            )
        if 'odom_timeout_sec' in ros_parameters:
            self.odom_timeout_sec = float(
                ros_parameters['odom_timeout_sec']
            )
        if 'odom_forward_distance_m' in ros_parameters:
            self.odom_forward_distance_m = float(
                ros_parameters['odom_forward_distance_m']
            )
        if 'odom_backward_distance_m' in ros_parameters:
            self.odom_backward_distance_m = float(
                ros_parameters['odom_backward_distance_m']
            )
        if 'odom_turn_distance_m' in ros_parameters:
            self.odom_turn_distance_m = float(
                ros_parameters['odom_turn_distance_m']
            )
        if 'straight_correction_angular_z' in ros_parameters:
            self.straight_correction_angular_z = float(
                ros_parameters['straight_correction_angular_z']
            )
        if 'sequence_1_forward_start_distance_m' in ros_parameters:
            self.sequence_1_forward_start_distance_m = float(
                ros_parameters['sequence_1_forward_start_distance_m']
            )
        if 'sequence_1_j_turn_1_distance_m' in ros_parameters:
            self.sequence_1_j_turn_1_distance_m = float(
                ros_parameters['sequence_1_j_turn_1_distance_m']
            )
        if 'sequence_1_backward_distance_m' in ros_parameters:
            self.sequence_1_backward_distance_m = float(
                ros_parameters['sequence_1_backward_distance_m']
            )
        if 'sequence_1_j_turn_2_distance_m' in ros_parameters:
            self.sequence_1_j_turn_2_distance_m = float(
                ros_parameters['sequence_1_j_turn_2_distance_m']
            )
        if 'sequence_1_backward_2_distance_m' in ros_parameters:
            self.sequence_1_backward_2_distance_m = float(
                ros_parameters['sequence_1_backward_2_distance_m']
            )
        if 'sequence_1_j_turn_3_distance_m' in ros_parameters:
            self.sequence_1_j_turn_3_distance_m = float(
                ros_parameters['sequence_1_j_turn_3_distance_m']
            )
        if 'sequence_2_forward_start_distance_m' in ros_parameters:
            self.sequence_2_forward_start_distance_m = float(
                ros_parameters['sequence_2_forward_start_distance_m']
            )
        if 'sequence_2_k_turn_1_distance_m' in ros_parameters:
            self.sequence_2_k_turn_1_distance_m = float(
                ros_parameters['sequence_2_k_turn_1_distance_m']
            )
        if 'sequence_2_backward_distance_m' in ros_parameters:
            self.sequence_2_backward_distance_m = float(
                ros_parameters['sequence_2_backward_distance_m']
            )
        if 'sequence_2_k_turn_2_distance_m' in ros_parameters:
            self.sequence_2_k_turn_2_distance_m = float(
                ros_parameters['sequence_2_k_turn_2_distance_m']
            )
        if 'sequence_2_backward_2_distance_m' in ros_parameters:
            self.sequence_2_backward_2_distance_m = float(
                ros_parameters['sequence_2_backward_2_distance_m']
            )
        if 'sequence_2_k_turn_3_distance_m' in ros_parameters:
            self.sequence_2_k_turn_3_distance_m = float(
                ros_parameters['sequence_2_k_turn_3_distance_m']
            )
        if 'odom_segment_timeout_sec' in ros_parameters:
            self.odom_segment_timeout_sec = float(
                ros_parameters['odom_segment_timeout_sec']
            )
        if 'odom_stop_buffer_m' in ros_parameters:
            self.odom_stop_buffer_m = float(
                ros_parameters['odom_stop_buffer_m']
            )
        if 'odom_speed_profile_enabled' in ros_parameters:
            self.odom_speed_profile_enabled = bool(
                ros_parameters['odom_speed_profile_enabled']
            )
        if 'odom_slowdown_distance_m' in ros_parameters:
            self.odom_slowdown_distance_m = float(
                ros_parameters['odom_slowdown_distance_m']
            )
        if 'odom_min_speed_scale' in ros_parameters:
            self.odom_min_speed_scale = float(
                ros_parameters['odom_min_speed_scale']
            )
        if 'odom_min_linear_speed' in ros_parameters:
            self.odom_min_linear_speed = float(
                ros_parameters['odom_min_linear_speed']
            )
        if 'use_vesc_state_distance_fallback' in ros_parameters:
            self.use_vesc_state_distance_fallback = bool(
                ros_parameters['use_vesc_state_distance_fallback']
            )
        if 'vesc_state_topic' in ros_parameters:
            self.vesc_state_topic = str(
                ros_parameters['vesc_state_topic']
            )
        if 'vesc_state_timeout_sec' in ros_parameters:
            self.vesc_state_timeout_sec = float(
                ros_parameters['vesc_state_timeout_sec']
            )
        if 'vesc_speed_to_erpm_gain' in ros_parameters:
            self.vesc_speed_to_erpm_gain = float(
                ros_parameters['vesc_speed_to_erpm_gain']
            )
        if 'vesc_speed_to_erpm_offset' in ros_parameters:
            self.vesc_speed_to_erpm_offset = float(
                ros_parameters['vesc_speed_to_erpm_offset']
            )
        if 'auto_save_debounce_sec' in ros_parameters:
            self.auto_save_debounce_sec = float(
                ros_parameters['auto_save_debounce_sec']
            )
        if 'opencv_camera_source' in ros_parameters:
            self.opencv_camera_source = str(
                ros_parameters['opencv_camera_source']
            )
        if 'opencv_camera_search_indices' in ros_parameters:
            self.opencv_camera_search_indices = str(
                ros_parameters['opencv_camera_search_indices']
            )
        if 'opencv_camera_index' in ros_parameters:
            self.opencv_camera_index = int(
                ros_parameters['opencv_camera_index']
            )
        if 'bev_enabled' in ros_parameters:
            self.bev_enabled = bool(ros_parameters['bev_enabled'])
        if 'bev_width' in ros_parameters:
            self.bev_width = int(ros_parameters['bev_width'])
        if 'bev_height' in ros_parameters:
            self.bev_height = int(ros_parameters['bev_height'])
        if 'approach_trigger_enabled' in ros_parameters:
            self.approach_trigger_enabled = bool(
                ros_parameters['approach_trigger_enabled']
            )
        if 'approach_roi_enabled' in ros_parameters:
            self.approach_roi_enabled = bool(
                ros_parameters['approach_roi_enabled']
            )
        if 'approach_roi_x_min' in ros_parameters:
            self.approach_roi_x_min = int(
                ros_parameters['approach_roi_x_min']
            )
        if 'approach_roi_y_min' in ros_parameters:
            self.approach_roi_y_min = int(
                ros_parameters['approach_roi_y_min']
            )
        if 'approach_roi_x_max' in ros_parameters:
            self.approach_roi_x_max = int(
                ros_parameters['approach_roi_x_max']
            )
        if 'approach_roi_y_max' in ros_parameters:
            self.approach_roi_y_max = int(
                ros_parameters['approach_roi_y_max']
            )
        if 'approach_min_component_area' in ros_parameters:
            self.approach_min_component_area = int(
                ros_parameters['approach_min_component_area']
            )
        if 'approach_max_component_area' in ros_parameters:
            self.approach_max_component_area = int(
                ros_parameters['approach_max_component_area']
            )
        if 'approach_target_component_count' in ros_parameters:
            self.approach_target_component_count = int(
                ros_parameters['approach_target_component_count']
            )
        if 'approach_count_tolerance' in ros_parameters:
            self.approach_count_tolerance = int(
                ros_parameters['approach_count_tolerance']
            )
        if 'approach_required_stable_frames' in ros_parameters:
            self.approach_required_stable_frames = int(
                ros_parameters['approach_required_stable_frames']
            )
        if 'motor_calibration_enabled' in ros_parameters:
            self.motor_calibration_enabled = bool(
                ros_parameters['motor_calibration_enabled']
            )
        if 'motion_enabled' in ros_parameters:
            self.motion_enabled = bool(ros_parameters['motion_enabled'])
        if 'rc_control_enabled' in ros_parameters:
            self.rc_control_enabled = bool(
                ros_parameters['rc_control_enabled']
            )
        if 'rc_linear_step' in ros_parameters:
            self.rc_linear_step = float(ros_parameters['rc_linear_step'])
        if 'rc_angular_step' in ros_parameters:
            self.rc_angular_step = float(ros_parameters['rc_angular_step'])
        if 'rc_max_linear_speed' in ros_parameters:
            self.rc_max_linear_speed = float(
                ros_parameters['rc_max_linear_speed']
            )
        if 'rc_max_angular_z' in ros_parameters:
            self.rc_max_angular_z = float(
                ros_parameters['rc_max_angular_z']
            )
        if 'test_linear_speed' in ros_parameters:
            self.test_linear_speed = float(
                ros_parameters['test_linear_speed']
            )
        if 'test_backward_speed' in ros_parameters:
            self.test_backward_speed = float(
                ros_parameters['test_backward_speed']
            )
        if 'test_angular_left' in ros_parameters:
            self.test_angular_left = float(
                ros_parameters['test_angular_left']
            )
        if 'test_angular_right' in ros_parameters:
            self.test_angular_right = float(
                ros_parameters['test_angular_right']
            )
        if 'test_pulse_duration_sec' in ros_parameters:
            self.test_pulse_duration_sec = float(
                ros_parameters['test_pulse_duration_sec']
            )
        if 'debug_image_dir' in ros_parameters:
            self.debug_image_dir = Path(ros_parameters['debug_image_dir'])
            if not self.debug_image_dir.is_absolute():
                self.debug_image_dir = (
                    package_root_from_module() / self.debug_image_dir
                )

        self.pending_autosave = False
        self._write_trackbars()
        self.get_logger().info(f'Loaded parameters from {self.config_file}')

    def save_debug_images(self):
        if self.latest_original is None:
            self.get_logger().warning('No camera frame has been received yet.')
            return

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        self.debug_image_dir.mkdir(parents=True, exist_ok=True)
        image_set = {
            'original': self.latest_original,
            'blue_mask': self.latest_mask,
            'cleaned_mask': self.latest_cleaned_mask,
            'canny_edges': self.latest_edges,
            'debug_overlay': self.latest_overlay,
            'bev_image': self.latest_bev_image,
            'bev_mask': self.latest_bev_mask,
            'bev_line_overlay': self.latest_bev_line_overlay,
            'approach_roi_mask': self.latest_approach_roi_mask,
        }

        for name, image in image_set.items():
            output_path = self.debug_image_dir / f'{timestamp}_{name}.png'
            cv2.imwrite(str(output_path), image)

        self.get_logger().info(
            f'Saved debug image set to {self.debug_image_dir}'
        )

    def publish_zero_velocity(self):
        if not rclpy.ok():
            return

        twist = Twist()
        try:
            self.cmd_vel_publisher.publish(twist)
        except Exception as error:
            self.get_logger().debug(f'Zero velocity publish skipped: {error}')

        if not self.publish_ackermann:
            return

        ackermann = AckermannDriveStamped()
        ackermann.header.stamp = self.get_clock().now().to_msg()
        try:
            self.ackermann_publisher.publish(ackermann)
        except Exception as error:
            self.get_logger().debug(
                f'Zero Ackermann publish skipped: {error}'
            )

    def request_shutdown(self):
        if self.shutdown_requested:
            return

        self.shutdown_requested = True
        self.publish_zero_velocity()
        self._release_opencv_camera()
        self.close_windows()
        self.get_logger().info('Quit requested. Shutting down safely.')
        rclpy.shutdown()

    def close_windows(self):
        if not self.gui_available:
            return

        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        self.gui_available = False

    def _warn_throttled(self, message, throttle_period=5.0):
        now = time.monotonic()
        if now - self.last_camera_warning_time < throttle_period:
            return

        self.last_camera_warning_time = now
        self.get_logger().warning(message)

    def _image_topic_summary(self):
        publisher_count = self.count_publishers(self.camera_topic)
        try:
            image_topics = [
                name for name, topic_types in self.get_topic_names_and_types()
                if 'sensor_msgs/msg/Image' in topic_types
            ]
        except Exception:
            return 'Could not query ROS image topics.'

        if not image_topics:
            return (
                f'Publishers on {self.camera_topic}: {publisher_count}. '
                'No sensor_msgs/msg/Image topics are currently visible.'
            )

        return (
            f'Publishers on {self.camera_topic}: {publisher_count}. '
            f'Visible image topics: {", ".join(sorted(image_topics))}'
        )

    @staticmethod
    def _clamp(value, lower_bound, upper_bound):
        return max(lower_bound, min(int(value), upper_bound))

    @staticmethod
    def _clamp_float(value, lower_bound, upper_bound):
        return max(float(lower_bound), min(float(value), float(upper_bound)))


def main(args=None):
    rclpy.init(args=args)
    parking_node = ParkingNode()

    try:
        rclpy.spin(parking_node)
    except KeyboardInterrupt:
        if rclpy.ok():
            parking_node.get_logger().info('Keyboard interrupt received.')
    finally:
        if rclpy.ok() and not parking_node.shutdown_requested:
            parking_node.publish_zero_velocity()
        parking_node._release_opencv_camera()
        parking_node.close_windows()
        parking_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
