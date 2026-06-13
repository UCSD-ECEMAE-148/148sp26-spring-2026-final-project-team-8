import os
import select
import sys
import termios
import time
import tty

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class RcKeyboardNode(Node):
    def __init__(self):
        super().__init__('rc_keyboard_node')
        self.declare_parameter('keyboard_topic', '/parking_system/key')
        self.keyboard_topic = str(self.get_parameter('keyboard_topic').value)
        self.publisher = self.create_publisher(String, self.keyboard_topic, 10)

    def publish_key(self, key):
        message = String()
        message.data = key
        self.publisher.publish(message)
        self.get_logger().info(f'key: {repr(key)}')


def print_controls():
    print('')
    print('parking_system RC step keyboard')
    print('  e : motion enable/disable')
    print('  f : forward one odom step')
    print('  b : backward one odom step')
    print('  j : j-turn one odom step')
    print('  k : k-turn one odom step')
    print('  1 : sequence 1')
    print('  2 : sequence 2')
    print('  3 : reverse sequence 1')
    print('  4 : reverse sequence 2')
    print('  x : emergency stop')
    print('  q : quit parking_node')
    print('')
    print('Focus this terminal and press keys directly.')
    sys.stdout.flush()


def main(args=None):
    rclpy.init(args=args)
    node = RcKeyboardNode()
    tty_fd = None
    old_settings = None

    try:
        tty_fd = os.open('/dev/tty', os.O_RDONLY | os.O_NONBLOCK)
        old_settings = termios.tcgetattr(tty_fd)
        tty.setcbreak(tty_fd)
        print_controls()

        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)
            readable, _, _ = select.select([tty_fd], [], [], 0.05)
            if not readable:
                continue

            data = os.read(tty_fd, 1)
            if not data:
                continue
            if data == b'\x03':
                raise KeyboardInterrupt

            key = data.decode('utf-8', errors='ignore')
            if not key:
                continue

            node.publish_key(key)
            if key == 'q':
                time.sleep(0.1)
                break
    except KeyboardInterrupt:
        pass
    except OSError as error:
        node.get_logger().error(f'Could not read /dev/tty: {error}')
    finally:
        if old_settings is not None and tty_fd is not None:
            termios.tcsetattr(tty_fd, termios.TCSADRAIN, old_settings)
        if tty_fd is not None:
            os.close(tty_fd)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
