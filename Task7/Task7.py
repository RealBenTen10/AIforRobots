from numpy import cos, sin
from matplotlib import pyplot as plt
import numpy as np

drive_in_circles = False
noise_strength = 0.0005
state = [0.0, 0.0, 0.0]
optimal_state = [0.0, 0.0, 0.0]
velocity = [1.0, 1.0]
axisLength = 1
steps_per_run = 10
steps = 1000
x_states = []
y_states = []


def get_velocity():
    noise = np.random.normal(0, noise_strength, 2)
    print([velocity[0] + noise[0], velocity[1] + noise[1]])
    return velocity[0] + noise[0], velocity[1] + noise[1]


def get_circular_velocity():
    left_wheel = velocity[0] + 0.2
    right_wheel = velocity[1] + 0.2
    noise = np.random.normal(0, noise_strength, 2)
    return left_wheel + noise[0], right_wheel + noise[1]


def update_state(state, v_l, v_r, axisLength):
    deltaX = 0.5 * cos(state[2]) * v_l + 0.5 * cos(state[2]) * v_r
    deltaY = 0.5 * sin(state[2]) * v_l + 0.5 * sin(state[2]) * v_r
    deltaTheta = 1 / axisLength * (v_l - v_r)
    state[0] += deltaX
    state[1] += deltaY
    state[2] += deltaTheta
    optimal_state[0] += 1


def compute_odometry():
    odometry_state = [abs(state[0] - optimal_state[0]), abs(state[1] - optimal_state[1]),
                      abs(state[2] - optimal_state[2])]
    return odometry_state


def simulate(drive_in_circles):
    state = [0.0, 0.0, 0.0]
    optimal_state = [0.0, 0.0, 0.0]
    x_states = []
    y_states = []

    for i in range(steps):
        for j in range(steps_per_run):
            if drive_in_circles:
                v_l, v_r = get_circular_velocity()
            else:
                v_l, v_r = get_velocity()
            update_state(state, v_l, v_r, axisLength)
            optimal_state[0] += 1
        odom_x = abs(state[0] - optimal_state[0])
        odom_y = abs(state[1] - optimal_state[1])
        x_states.append(odom_x)
        y_states.append(odom_y)

    return x_states, y_states


x_straight, y_straight = simulate(drive_in_circles=False)
x_circle, y_circle = simulate(drive_in_circles=True)

H1, xEdges1, yEdges1 = np.histogram2d(x_straight, y_straight, bins=10)
H2, xEdges2, yEdges2 = np.histogram2d(x_circle, y_circle, bins=10)
H1 = np.flipud(np.rot90(H1))
H2 = np.flipud(np.rot90(H2))
H1_masked = np.ma.masked_where(H1 == 0, H1)
H2_masked = np.ma.masked_where(H2 == 0, H2)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.title("Straight Line")
plt.pcolormesh(xEdges1, yEdges1, H1_masked, cmap='Blues', shading='auto')
plt.colorbar(label='Counts')

plt.subplot(1, 2, 2)
plt.title("Circle")
plt.pcolormesh(xEdges2, yEdges2, H2_masked, cmap='Reds', shading='auto')
plt.colorbar(label='Counts')

plt.tight_layout()
plt.show()