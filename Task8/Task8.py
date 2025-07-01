import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import math
import numbers

threshold = 1e-9  # tolerance for line fit deviation


# Linear regression over a set of points
def linear_regression(points):
    if len(points) < 2:
        return None, None
    X = points[:, 0].reshape(-1, 1)
    y = points[:, 1]
    model = LinearRegression().fit(X, y)
    m = model.coef_[0]
    b = model.intercept_
    return m, b


# Find the point with maximum perpendicular distance to the line y = mx + b
def find_furthest_point(points, m, b):
    if len(points) < 2:
        return None, None
    distances = np.abs(points[:, 1] - (m * points[:, 0] + b))
    index = np.argmax(distances)
    distance = distances[index]
    return index, distance



# Recursive split-and-merge line segmentation
def compute_line_map(points, threshold):
    segments = []

    # Step 1: Fit a line to all points
    m, b = linear_regression(points)

    # Step 2: Find the point with the maximum distance to the line
    index, distance = find_furthest_point(points, m, b)

    if distance is not None and distance > threshold:
        # Step 3: Use a new line from the first to the last point to decide where to split
        first_last_line = linear_regression(np.array([points[0], points[-1]]))
        split_index, _ = find_furthest_point(points, *first_last_line)

        # Split at the furthest point from the first-to-last line
        left = points[:split_index + 1]
        right = points[split_index:]

        if len(left) < 2 or len(right) < 2:
            # Stop if splitting fails
            segments.append((points, m, b))
        else:
            # Recurse on both subsets
            segments += compute_line_map(left, threshold)
            segments += compute_line_map(right, threshold)
    else:
        # Accept this segment
        segments.append((points, m, b))

    return segments


# Example point cloud

points = np.loadtxt("/home/ben/PycharmProjects/AIforRobots/Task8/data.txt", delimiter=",")


def get_cartesian_coordinates(ranges):
    angle = 0
    step = 360 / len(ranges)
    coordinates = []
    for distance in ranges:
        if isinstance(distance, numbers.Number):
            theta_rad = math.radians(angle)
            x = distance * math.cos(theta_rad)
            y = distance * math.sin(theta_rad)
            coordinates.append([x, y])
        else:
            coordinates.append([np.inf, np.inf])
        angle += step
    return coordinates
temp_coordinates = []
for point in points:
    temp_coordinates.append(get_cartesian_coordinates(point))
coordinates = []
for i in range(len(temp_coordinates[0])):
    for j in range(len(temp_coordinates)):
        if not math.isinf(temp_coordinates[j][i][0]):
            coordinates.append(temp_coordinates[j][i])
coordinates = np.array(coordinates)
# Compute line map
print("doing something")
segments = compute_line_map(coordinates, threshold)

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Scatter original points
ax1.scatter(coordinates[:, 0], coordinates[:, 1], s=20, alpha=0.3)
ax1.set_title("Raw Points")

# Scatter with segment coloring
colors = plt.cm.tab10(np.linspace(0, 1, len(segments)))
for i, (seg_points, m, b) in enumerate(segments):
    ax1.scatter(seg_points[:, 0], seg_points[:, 1], s=20, alpha=0.8, color=colors[i], label=f"Segment {i+1}")
ax1.set_title("Segmented Points")

# Plot regression lines
ax2.scatter(points[:, 0], points[:, 1], s=20, alpha=0.3)
for i, (seg_points, m, b) in enumerate(segments):
    x_vals = np.linspace(np.min(seg_points[:, 0]), np.max(seg_points[:, 0]), 100)
    y_vals = m * x_vals + b
    ax2.plot(x_vals, y_vals, color=colors[i], label=f"Line {i+1}")
ax2.set_title("Line Map")


plt.tight_layout()
plt.show()
