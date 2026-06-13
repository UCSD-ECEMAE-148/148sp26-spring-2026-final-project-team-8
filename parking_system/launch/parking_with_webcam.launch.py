from parking_system.launch_helpers import generate_parking_camera_launch


def generate_launch_description():
    print(
        'parking_system: parking_with_webcam.launch.py is compatibility '
        'mode; selecting the active camera from car_config.yaml.'
    )
    return generate_parking_camera_launch()
