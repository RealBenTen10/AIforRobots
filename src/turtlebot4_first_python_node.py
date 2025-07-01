from time import sleep

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
import numpy as np
from sklearn.linear_model import LinearRegression


class ScanSubscriber(Node):
    def __init__(self):
        super().__init__('scan_subscriber')

        qos_profile = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT
        )

        self.subscription = self.create_subscription(
            LaserScan,
            '/turtlebot4_3/scan',
            self.listener_callback,
            qos_profile
        )
        self.threshold = 0.0001
        self.data = []
        self.coordinates = []
        self.points = []

    def listener_callback(self, msg: LaserScan):
        ''' Get the ranges from the scan '''
        # self.get_logger().info(f"Received {len(msg.ranges)} ranges: {msg.ranges}")
        if (len(self.data)) < 11:
            test = np.array(msg.ranges)
            print(test)
            self.data.append(test)
            print("Got some data - sleeping now")
            sleep(2)
        else:
            self.get_plot()
            sleep(100)


    def get_cartesian_coordinates(self):
        ''' Calculate the cartesian coordinates from the scan data '''
        temp_coordinates = []
        for ranges in self.data:
            angle = 0
            step = 360 / len(ranges)
            print(step)
            for distance in ranges:
                if isinstance(distance, numbers.Number):
                    theta_rad = math.radians(angle)
                    x = distance * math.cos(theta_rad)
                    y = distance * math.sin(theta_rad)
                    print(f"angle: {angle} , x: {x} , y: {y}")
                    temp_coordinates.append([x, y])
                angle += step
        self.coordinates.append(temp_coordinates)

    def linear_regression(self):
        if len(points) < 2:
            return None, None, None
        X = self.points[:, 0].reshape(-1, 1)
        y = self.points[:, 1]
        model = LinearRegression().fit(X, y)
        m = model.coef_[0]
        b = model.intercept_
        return m, b

    def find_furthest_point(self, m, b):
        if len(points) < 2:
            return None, None
        distances = np.abs(self.points[:, 1] - (m * self.points[:, 0] + b))
        furthest_index = np.argmax(distances)
        furthest_distance = distances[furthest_index]
        print(f"Furthest point index: {furthest_index}, distance: {furthest_distance}")
        return furthest_index, furthest_distance

    def compute_line_map(self):
        segments = []
        points_temp = []
        for index in range(len(self.coordinates[0])):
            for index_j in range(len(self.coordinates)):
                points_temp.append(self.coordinates[index][index_j])
        self.points = points_temp

        # Step 1: Fit a line to all points
        m, b = self.linear_regression()

        # Step 2: Find the point with the maximum distance to the line
        index, distance = find_furthest_point(points, m, b)

        if distance is not None and distance > self.threshold:
            # Step 3: Use a new line from the first to the last point to decide where to split
            first_last_line = linear_regression(np.array([points[0], points[-1]]))
            split_index, _ = find_furthest_point(points, *first_last_line)

            # Split at the furthest point from the first-to-last line
            left = points[:split_index + 1]
            right = points[split_index:]
            print("left and right: ", left, right)

            if len(left) < 2 or len(right) < 2:
                # Stop if splitting fails
                segments.append((points, m, b))
            else:
                # Recurse on both subsets
                segments += compute_line_map(left, self.threshold)
                segments += compute_line_map(right, self.threshold)
        else:
            # Accept this segment
            segments.append((points, m, b))

        return segments

    def get_plot(self):

        # Compute line map
        segments = self.compute_line_map()

        # Plotting
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Scatter original points
        ax1.scatter(points[:, 0], points[:, 1], s=20, alpha=0.3)
        ax1.set_title("Raw Points")

        # Scatter with segment coloring
        colors = plt.cm.tab10(np.linspace(0, 1, len(segments)))
        for i, (seg_points, m, b) in enumerate(segments):
            ax1.scatter(seg_points[:, 0], seg_points[:, 1], s=20, alpha=0.8, color=colors[i], label=f"Segment {i + 1}")
        ax1.legend()
        ax1.set_title("Segmented Points")

        # Plot regression lines
        ax2.scatter(points[:, 0], points[:, 1], s=20, alpha=0.3)
        for i, (seg_points, m, b) in enumerate(segments):
            x_vals = np.linspace(np.min(seg_points[:, 0]), np.max(seg_points[:, 0]), 100)
            y_vals = m * x_vals + b
            ax2.plot(x_vals, y_vals, color=colors[i], label=f"Line {i + 1}")
        ax2.set_title("Line Map")
        ax2.legend()

        plt.tight_layout()
        plt.show()




def main(args=None):
    rclpy.init(args=args)
    node = ScanSubscriber()
    print("Starting Node")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':

    main()
