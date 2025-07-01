import random
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import math

TILES = ["white", "black"]
ACTIONS = ["left", "right"]
string_white = "white"
string_black = "black"
string_left = "left"
string_right = "right"
line_break = "-" * 100
n_steps = 100
world_size = 10  # Number of tiles in 1D world

# Simulate noisy color observation
def noisy_observation(true_color, noise=0.0):
    if random.random() < noise:
        return string_black if true_color == string_white else string_white
    return true_color

# Simulate noisy movement
def noisy_movement(chosen_action, noise=0.0):
    if random.random() < noise:
        return string_left if chosen_action == string_right else string_right
    return chosen_action

class Robot:
    def __init__(self, strategy="adventurous"):
        self.position = math.floor(world_size/2)
        self.world = [string_white if i % 2 == 0 else string_black for i in range(world_size)]
        self.histograms = [defaultdict(int) for _ in range(world_size)]
        self.strategy = strategy
        print(f"Robot started at position {self.position} ({self.world[self.position]})")
        print(line_break)

    def perceive_and_update(self, noise=0.0):
        observed_color = noisy_observation(self.world[self.position], noise)
        self.histograms[self.position][observed_color] += 1

    def get_position_stats(self):
        stats = []
        for hist in self.histograms:
            white = hist[string_white]
            black = hist[string_black]
            total = white + black
            if total == 0:
                mean = 0.5
                var = 0.25  # Max variance for binomial
            else:
                p_white = white / total
                mean = p_white
                var = p_white * (1 - p_white)
            stats.append((mean, var))
        return stats

    def choose_next_position(self):
        stats = self.get_position_stats()
        certainties = []
        for i, (mean, var) in enumerate(stats):
            if self.strategy == "adventurous":
                certainties.append((1 - var, i))  # most certain
            elif self.strategy == "cautious":
                certainties.append((var, i))  # most uncertain
        certainties.sort(reverse=True)
        best_pos = certainties[0][1]
        if best_pos < self.position:
            return string_left
        elif best_pos > self.position:
            return string_right
        else:
            return random.choice(ACTIONS)  # Stay or random move

    def move(self, action, noise=0.0):
        final_action = noisy_movement(action, noise)
        if final_action == string_left:
            self.position = max(0, self.position - 1)
        elif final_action == string_right:
            self.position = min(world_size - 1, self.position + 1)

    def step(self, perception_noise=0.0, movement_noise=0.0):
        print(f"Position {self.position}, True: {self.world[self.position]}")
        self.perceive_and_update(perception_noise)
        action = self.choose_next_position()
        self.move(action, movement_noise)
        print(f"Moved {action} to position {self.position}")
        print(line_break)

    def visualize_histograms(self):
        white_probs = []
        black_probs = []
        for hist in self.histograms:
            white = hist[string_white] + 1
            black = hist[string_black] + 1
            total = white + black
            if total == 0:
                white_probs.append(0.5)
                black_probs.append(0.5)
            else:
                white_probs.append(white / total)
                black_probs.append(black / total)

        x = list(range(world_size))
        plt.figure(figsize=(10, 4))
        plt.bar(x, white_probs, color="gray", label="White")
        plt.bar(x, black_probs, bottom=white_probs, color="black", label="Black")
        plt.xlabel("Tile Position")
        plt.ylabel("Estimated Probabilities")
        plt.title("Robot's World Model")
        plt.legend()
        plt.show()


# Run both strategies
print("=== Cautious Strategy ===")
robot = Robot(strategy="adventurous")
for _ in range(n_steps):
    robot.step()
robot.visualize_histograms()

print("\n=== Adventurous Strategy ===")
robot_cautious = Robot(strategy="cautious")
for _ in range(n_steps):
    robot_cautious.step(0.2, 0.2)
robot_cautious.visualize_histograms()
