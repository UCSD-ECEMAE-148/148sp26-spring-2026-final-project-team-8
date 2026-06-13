import enum

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger


class State(enum.Enum):
    ENABLING      = "ENABLING"      # waiting to send m/e to unlock motors
    SCANNING      = "SCANNING"
    ADVANCING     = "ADVANCING"
    PARKING_LEFT  = "PARKING_LEFT"
    PARKING_RIGHT = "PARKING_RIGHT"
    PARKED        = "PARKED"


# How long to wait after startup before sending the motor-enable keys.
# parking_node needs time to finish initializing inside parking_with_odom.launch.py.
MOTOR_ENABLE_DELAY_SEC = 8.0

# How long to wait after sending "f" before assuming the forward move finished.
# Tuned to odom_forward_distance_m / odom_min_linear_speed + margin.
FORWARD_WAIT_SEC = 5.0

# How long to wait after sending "1" or "2" for the full parking sequence to finish.
# Should be >= sum of all odom segment timeouts in the sequence.
PARKING_WAIT_SEC = 30.0

TICK_PERIOD = 0.1


class ParkingControllerNode(Node):
    def __init__(self):
        super().__init__("parking_controller_node")

        self._state = State.ENABLING
        self._state_start = self.get_clock().now()
        self._cmd_sent = False      # ensures we publish each key command only once
        self._scan_pending = False

        self._key_pub = self.create_publisher(String, "/parking_system/key", 10)
        self._state_pub = self.create_publisher(String, "/controller_state", 10)
        self._scan_client = self.create_client(Trigger, "/request_scan")

        self._timer = self.create_timer(TICK_PERIOD, self._tick)
        self.get_logger().info(
            f"ParkingControllerNode ready — enabling motors in {MOTOR_ENABLE_DELAY_SEC:.0f}s"
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _elapsed(self):
        return (self.get_clock().now() - self._state_start).nanoseconds / 1e9

    def _set_state(self, new_state: State):
        self.get_logger().info(f"{self._state.value} -> {new_state.value}")
        self._state = new_state
        self._state_start = self.get_clock().now()
        self._cmd_sent = False

    def _send_key(self, key: str):
        msg = String()
        msg.data = key
        self._key_pub.publish(msg)
        self.get_logger().info(f"Sent command: '{key}'")

    # ── state machine ─────────────────────────────────────────────────────────

    def _tick(self):
        state_msg = String()
        state_msg.data = self._state.value
        self._state_pub.publish(state_msg)

        if self._state == State.ENABLING:
            self._handle_enabling()
        elif self._state == State.SCANNING:
            self._handle_scanning()
        elif self._state == State.ADVANCING:
            self._handle_advancing()
        elif self._state == State.PARKING_LEFT:
            self._handle_parking("1", PARKING_WAIT_SEC)
        elif self._state == State.PARKING_RIGHT:
            self._handle_parking("2", PARKING_WAIT_SEC)
        elif self._state == State.PARKED:
            pass  # parking_node already stopped the car at end of sequence

    def _handle_enabling(self):
        if self._elapsed() < MOTOR_ENABLE_DELAY_SEC:
            self.get_logger().info(
                f"Waiting for parking_node... "
                f"{MOTOR_ENABLE_DELAY_SEC - self._elapsed():.0f}s remaining",
                throttle_duration_sec=2,
            )
            return
        if not self._cmd_sent:
            # m and e both default to false in parking_params.yaml, so one
            # publish each toggles them on deterministically at startup.
            self._send_key("m")
            self._send_key("e")
            self._cmd_sent = True
        self._set_state(State.SCANNING)

    def _handle_scanning(self):
        if self._scan_pending:
            return
        if not self._scan_client.service_is_ready():
            self.get_logger().info(
                "Waiting for /request_scan service...", throttle_duration_sec=5
            )
            return
        self._scan_pending = True
        future = self._scan_client.call_async(Trigger.Request())
        future.add_done_callback(self._scan_done)
        self.get_logger().info("Scan requested")

    def _scan_done(self, future):
        self._scan_pending = False
        try:
            result = future.result()
        except Exception as exc:
            self.get_logger().error(f"Scan call failed: {exc} — retrying")
            self._set_state(State.SCANNING)
            return

        direction = result.message
        self.get_logger().info(f"Scan result: '{direction}'")

        if direction == "left":
            self._set_state(State.PARKING_LEFT)
        elif direction == "right":
            self._set_state(State.PARKING_RIGHT)
        else:
            self._set_state(State.ADVANCING)

    def _handle_advancing(self):
        if not self._cmd_sent:
            self._send_key("f")
            self._cmd_sent = True
        if self._elapsed() >= FORWARD_WAIT_SEC:
            self._set_state(State.SCANNING)

    def _handle_parking(self, key: str, wait_sec: float):
        if not self._cmd_sent:
            self._send_key(key)
            self._cmd_sent = True
        if self._elapsed() >= wait_sec:
            self.get_logger().info("Parking sequence complete")
            self._set_state(State.PARKED)

    def destroy_node(self):
        self._send_key("x")   # emergency stop on shutdown
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ParkingControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
