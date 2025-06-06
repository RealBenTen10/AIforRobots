import rclpy
from rclpy.node import Node
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import threading
import time

from std_msgs.msg import Header
from sensor_msgs.msg import Range  # If your topic is a custom msg, adjust import accordingly
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

# === Beam-based model parameters (can be adjusted) ===
Z_HIT = 0.8
Z_RAND = 0.2
SIGMA_HIT = 5.0  # Standard deviation for hit model
MAX_RANGE = 100.0  # Maximum range of sensor

class IRListener(Node):
    def __init__(self):
        super().__init__('ir_listener')
        qos_profile = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.subscription = self.create_subscription(
            Range,  # Or replace with actual message type
            '/ir_intensity/',
            self.listener_callback,
            qos_profile
        )
        self.buffer = deque(maxlen=640)  # Last ~10 seconds at 64 Hz
        self.all_data = []
        self.last_update_time = time.time()
        self.lock = threading.Lock()
        self.timer = self.create_timer(2.0, self.update_plot)

        # Set up plot
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.hist_plot = None
        self.model_plot = None

    def listener_callback(self, msg):
        try:
            # Extract the 'value' field from the message
            value = msg.readings[0].value  # Adjust if your message structure is different
            with self.lock:
                self.buffer.append(value)
                self.all_data.append(value)
        except Exception as e:
            self.get_logger().warn(f"Error parsing message: {e}")

    def update_plot(self):
        with self.lock:
            data = list(self.buffer)
            all_data = list(self.all_data)

        if len(data) == 0:
            return

        self.ax.clear()
        self.ax.set_title("IR Intensity Histogram (Last ~10s)")
        self.ax.set_xlabel("Intensity")
        self.ax.set_ylabel("Frequency")

        # Plot histogram
        counts, bins, _ = self.ax.hist(data, bins=range(0, int(MAX_RANGE)+5, 5), density=True, alpha=0.6, label="Measured Data")

        # Beam model probabilities
        x = np.linspace(0, MAX_RANGE, 100)
        p_hit = Z_HIT * (1.0 / (np.sqrt(2 * np.pi) * SIGMA_HIT)) * np.exp(-0.5 * ((x - 15) / SIGMA_HIT) ** 2)  # assuming peak at 15
        p_rand = Z_RAND * (1.0 / MAX_RANGE)
        p = p_hit + p_rand

        self.ax.plot(x, p, label="Beam Model", color="red")
        self.ax.legend()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

def main(args=None):
    rclpy.init(args=args)
    ir_listener = IRListener()

    # Start ROS 2 spin in a separate thread
    spin_thread = threading.Thread(target=rclpy.spin, args=(ir_listener,), daemon=True)
    spin_thread.start()

    try:
        while rclpy.ok():
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass

    ir_listener.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
