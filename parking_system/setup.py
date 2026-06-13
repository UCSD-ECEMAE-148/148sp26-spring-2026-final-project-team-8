from glob import glob
import os

from setuptools import setup


package_name = 'parking_system'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'),
         glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'),
         glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='djnighti@ucsd.edu',
    description='Camera-based parking tape HSV tuning and debug package.',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'parking_node = parking_system.parking_node:main',
            'rc_keyboard_node = parking_system.rc_keyboard_node:main',
            'oak_detection_node = parking_system.oak_detection_node:main',
            'parking_controller_node = parking_system.parking_controller_node:main',
        ],
    },
)
