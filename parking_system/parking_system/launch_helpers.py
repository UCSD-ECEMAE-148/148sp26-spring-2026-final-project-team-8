from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.actions import TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os
import yaml


CAMERA_ORDER = ('oakd', 'intel', 'webcam')
ACTUATOR_ORDER = ('vesc_without_odom', 'adafruit')


def active_camera_key():
    config = load_car_config()

    for camera_key in CAMERA_ORDER:
        if config.get(camera_key, 0) == 1:
            return camera_key

    return 'oakd'


def active_actuator_key():
    config = load_car_config()

    for actuator_key in ACTUATOR_ORDER:
        if config.get(actuator_key, 0) == 1:
            return actuator_key

    return 'vesc_without_odom'


def load_car_config():
    nav_share_dir = get_package_share_directory('ucsd_robocar_nav2_pkg')
    car_config = os.path.join(nav_share_dir, 'config', 'car_config.yaml')

    try:
        with open(car_config, 'r', encoding='utf-8') as config_file:
            return yaml.safe_load(config_file) or {}
    except OSError:
        return {}


def camera_launch_file(camera_key):
    launch_files = {
        'oakd': 'camera_oakd.launch.py',
        'intel': 'camera_intel.launch.py',
        'webcam': 'camera_webcam.launch.py',
    }
    return launch_files[camera_key]


def actuator_launch_file(actuator_key):
    launch_files = {
        'vesc_without_odom': 'vesc_twist.launch.py',
        'adafruit': 'adafruit_twist.launch.py',
    }
    return launch_files[actuator_key]


def generate_parking_camera_launch():
    parking_package = 'parking_system'
    nav_package = 'ucsd_robocar_nav2_pkg'

    camera_topic = LaunchConfiguration(
        'camera_topic',
        default='/camera/color/image_raw',
    )
    cmd_vel_topic = LaunchConfiguration(
        'cmd_vel_topic',
        default='/cmd_vel',
    )

    parking_config = os.path.join(
        get_package_share_directory(parking_package),
        'config',
        'parking_params.yaml',
    )

    all_components_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory(nav_package),
                'launch',
                'all_components.launch.py',
            )
        ),
    )

    parking_node = Node(
        package=parking_package,
        executable='parking_node',
        output='screen',
        parameters=[
            parking_config,
            {
                'odom_control_enabled': False,
                'rc_control_enabled': False,
            },
        ],
        remappings=[
            ('/camera/color/image_raw', camera_topic),
            ('/cmd_vel', cmd_vel_topic),
        ],
    )

    print('parking_system: starting UCSD all_components launch')

    return LaunchDescription([
        DeclareLaunchArgument(
            'camera_topic',
            default_value='/camera/color/image_raw',
            description='Image topic consumed by parking_node.',
        ),
        DeclareLaunchArgument(
            'cmd_vel_topic',
            default_value='/cmd_vel',
            description='Twist command topic published by parking_node.',
        ),
        all_components_launch,
        TimerAction(period=2.0, actions=[parking_node]),
    ])


def generate_parking_odom_launch():
    parking_package = 'parking_system'
    actuator_package = 'ucsd_robocar_actuator2_pkg'

    camera_topic = LaunchConfiguration(
        'camera_topic',
        default='/camera/color/image_raw',
    )
    cmd_vel_topic = LaunchConfiguration(
        'cmd_vel_topic',
        default='/cmd_vel',
    )
    ackermann_topic = LaunchConfiguration(
        'ackermann_topic',
        default='/drive',
    )
    ackermann_output_topic = LaunchConfiguration(
        'ackermann_output_topic',
        default='/ackermann_cmd',
    )
    odom_topic = LaunchConfiguration(
        'odom_topic',
        default='/odom',
    )
    vesc_config = LaunchConfiguration(
        'vesc_config',
        default=os.path.join(
            get_package_share_directory(parking_package),
            'config',
            'parking_vesc_odom.yaml',
        ),
    )

    parking_config = os.path.join(
        get_package_share_directory(parking_package),
        'config',
        'parking_params.yaml',
    )

    vesc_odom_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory(actuator_package),
                'launch',
                'vesc_odom.launch.py',
            )
        ),
        launch_arguments={
            'vesc_config': vesc_config,
            'topic_name': ackermann_output_topic,
        }.items(),
    )

    parking_node = Node(
        package=parking_package,
        executable='parking_node',
        output='screen',
        parameters=[
            parking_config,
            {
                'odom_control_enabled': True,
                'rc_control_enabled': False,
                'odom_topic': odom_topic,
                'ackermann_topic': ackermann_topic,
                'safety_timer_period': 0.02,
                'test_linear_speed': 0.306,
                'test_backward_speed': 0.306,
                'odom_min_linear_speed': 0.25,
                'perception_enabled': False,
                'motor_calibration_gui_only': True,
                'use_opencv_camera_fallback': False,
            },
        ],
        remappings=[
            ('/camera/color/image_raw', camera_topic),
            ('/cmd_vel', cmd_vel_topic),
        ],
    )

    print(
        'parking_system: starting vesc_odom with odom keyboard control '
        '(perception disabled)'
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'camera_topic',
            default_value='/camera/color/image_raw',
            description='Image topic consumed by parking_node.',
        ),
        DeclareLaunchArgument(
            'cmd_vel_topic',
            default_value='/cmd_vel',
            description='Twist fallback topic published by parking_node.',
        ),
        DeclareLaunchArgument(
            'ackermann_topic',
            default_value='/drive',
            description='Ackermann topic published by odom parking mode.',
        ),
        DeclareLaunchArgument(
            'ackermann_output_topic',
            default_value='/ackermann_cmd',
            description='Ackermann mux output topic consumed by VESC bridge.',
        ),
        DeclareLaunchArgument(
            'odom_topic',
            default_value='/odom',
            description='Odometry topic consumed by odom parking mode.',
        ),
        DeclareLaunchArgument(
            'vesc_config',
            default_value=os.path.join(
                get_package_share_directory(parking_package),
                'config',
                'parking_vesc_odom.yaml',
            ),
            description='VESC odom config used by parking odom mode.',
        ),
        vesc_odom_launch,
        TimerAction(period=2.0, actions=[parking_node]),
    ])


def generate_parking_rc_launch():
    parking_package = 'parking_system'
    actuator_package = 'ucsd_robocar_actuator2_pkg'

    cmd_vel_topic = LaunchConfiguration(
        'cmd_vel_topic',
        default='/cmd_vel',
    )
    ackermann_topic = LaunchConfiguration(
        'ackermann_topic',
        default='/drive',
    )
    ackermann_output_topic = LaunchConfiguration(
        'ackermann_output_topic',
        default='/ackermann_cmd',
    )
    keyboard_topic = LaunchConfiguration(
        'keyboard_topic',
        default='/parking_system/key',
    )
    vesc_config = LaunchConfiguration(
        'vesc_config',
        default=os.path.join(
            get_package_share_directory(parking_package),
            'config',
            'parking_vesc_odom.yaml',
        ),
    )

    parking_config = os.path.join(
        get_package_share_directory(parking_package),
        'config',
        'parking_params.yaml',
    )

    vesc_odom_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory(actuator_package),
                'launch',
                'vesc_odom.launch.py',
            )
        ),
        launch_arguments={
            'vesc_config': vesc_config,
            'topic_name': ackermann_output_topic,
        }.items(),
    )

    parking_node = Node(
        package=parking_package,
        executable='parking_node',
        output='screen',
        parameters=[
            parking_config,
            {
                'odom_control_enabled': True,
                'rc_control_enabled': False,
                'motor_calibration_enabled': True,
                'motion_enabled': False,
                'ackermann_topic': ackermann_topic,
                'odom_topic': '/odom',
                'safety_timer_period': 0.02,
                'test_linear_speed': 0.306,
                'test_backward_speed': 0.306,
                'odom_min_linear_speed': 0.25,
                'perception_enabled': False,
                'keyboard_topic_control_enabled': True,
                'keyboard_topic': keyboard_topic,
                'enable_gui': False,
                'motor_calibration_gui_only': True,
                'use_opencv_camera_fallback': False,
            },
        ],
        remappings=[
            ('/cmd_vel', cmd_vel_topic),
        ],
    )

    print(
        'parking_system: starting vesc_odom with 0.2m keyboard step control '
        '(perception disabled)'
    )
    print(
        'parking_system: open a second terminal and run: '
        'ros2 run parking_system rc_keyboard_node'
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'cmd_vel_topic',
            default_value='/cmd_vel',
            description='Twist fallback topic published by parking_node.',
        ),
        DeclareLaunchArgument(
            'ackermann_topic',
            default_value='/drive',
            description='Ackermann topic published by keyboard step mode.',
        ),
        DeclareLaunchArgument(
            'ackermann_output_topic',
            default_value='/ackermann_cmd',
            description='Ackermann mux output topic consumed by VESC bridge.',
        ),
        DeclareLaunchArgument(
            'vesc_config',
            default_value=os.path.join(
                get_package_share_directory(parking_package),
                'config',
                'parking_vesc_odom.yaml',
            ),
            description='VESC config used by keyboard step mode.',
        ),
        DeclareLaunchArgument(
            'keyboard_topic',
            default_value='/parking_system/key',
            description='Keyboard command topic used by parking_node.',
        ),
        vesc_odom_launch,
        TimerAction(period=2.0, actions=[parking_node]),
    ])


def generate_motor_calibration_launch():
    parking_package = 'parking_system'
    nav_package = 'ucsd_robocar_nav2_pkg'

    camera_topic = LaunchConfiguration(
        'camera_topic',
        default='/camera/color/image_raw',
    )
    cmd_vel_topic = LaunchConfiguration(
        'cmd_vel_topic',
        default='/cmd_vel',
    )

    parking_config = os.path.join(
        get_package_share_directory(parking_package),
        'config',
        'parking_params.yaml',
    )

    all_components_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory(nav_package),
                'launch',
                'all_components.launch.py',
            )
        ),
    )

    parking_node = Node(
        package=parking_package,
        executable='parking_node',
        output='screen',
        parameters=[
            parking_config,
            {
                'motor_calibration_gui_only': False,
                'use_opencv_camera_fallback': False,
                'odom_control_enabled': False,
                'rc_control_enabled': False,
            },
        ],
        remappings=[
            ('/camera/color/image_raw', camera_topic),
            ('/cmd_vel', cmd_vel_topic),
        ],
    )

    print('parking_system: starting UCSD all_components launch')

    return LaunchDescription([
        DeclareLaunchArgument(
            'camera_topic',
            default_value='/camera/color/image_raw',
            description='Image topic consumed by parking_node if available.',
        ),
        DeclareLaunchArgument(
            'cmd_vel_topic',
            default_value='/cmd_vel',
            description='Twist command topic used for motor calibration.',
        ),
        all_components_launch,
        TimerAction(period=2.0, actions=[parking_node]),
    ])


def generate_parking_auto_launch():
    parking_package = 'parking_system'

    # Start the full motor/odom stack (VESC driver + parking_node with odom control).
    # parking_params.yaml sets keyboard_topic_control_enabled: true so parking_node
    # listens on /parking_system/key for programmatic commands from parking_controller_node.
    parking_odom_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory(parking_package),
                'launch',
                'parking_with_odom.launch.py',
            )
        ),
    )

    oak_detection_node = Node(
        package=parking_package,
        executable='oak_detection_node',
        output='screen',
    )

    parking_controller_node = Node(
        package=parking_package,
        executable='parking_controller_node',
        output='screen',
    )

    print('parking_system: starting autonomous parking (OAK + Roboflow + controller)')

    return LaunchDescription([
        parking_odom_launch,
        TimerAction(period=2.0, actions=[oak_detection_node]),
        TimerAction(period=4.0, actions=[parking_controller_node]),
    ])
