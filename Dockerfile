FROM ghcr.io/ucsd-ecemae-148/ucsd_robocar:stable

WORKDIR /home/projects/ros2_ws/src
COPY parking_system/ parking_system/

WORKDIR /home/projects/ros2_ws
RUN bash -c "source /opt/ros/jazzy/setup.bash && colcon build --packages-select parking_system"

WORKDIR /home/projects/
