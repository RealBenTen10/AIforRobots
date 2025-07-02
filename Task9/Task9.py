from random import random, randint
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# possible colors for tiles
TILES = ["white", "black"]
# possible actions for robot
ACTIONS = ["left", "right"]

# New: Define a one-dimensional world with more than two tiles
# The world will be a list of colors, representing positions
# For example: ["white", "black", "white", "white", "black", "white", "black"]
WORLD_TILES = [
    "white",
    "black",
    "white",
    "white",
    "black",
    "white",
    "black"
]
WORLD_SIZE = len(WORLD_TILES)

# Line break for better readability
line_break = "-" * 50

# change this value for more (or less) steps
n_steps = 15


def get_tile_color(position):
    """Returns the color of the tile at a given position."""
    if 0 <= position < WORLD_SIZE:
        return WORLD_TILES[position]
    return None  # Indicate out of bounds


class Robot:
    def __init__(self):
        # self.world_model will store histograms for each position.
        # Each position's histogram will map "color" to "count".
        # Example: self.world_model[0] = {"white": 5, "black": 2}
        self.world_model = defaultdict(lambda: defaultdict(int))

        # Robot starts at a random valid position
        self.position = randint(0, WORLD_SIZE - 1)
        self.current_tile_color = get_tile_color(self.position)

        print(f"Robot started at position {self.position} on a {self.current_tile_color} tile.")
        print(line_break)

    def measure_tile_color(self):
        """Simulates the robot measuring the color of the current tile."""
        return get_tile_color(self.position)

    def update_world_model(self, position, color):
        """Updates the internal world model based on observation."""
        self.world_model[position][color] += 1

    def get_color_probabilities_at_position(self, position):
        """
        Computes the probabilities of colors at a given position based on the world model.
        Uses Laplace smoothing (+1) to avoid zero probabilities.
        """
        white_count = self.world_model[position]["white"] + 1
        black_count = self.world_model[position]["black"] + 1
        total = white_count + black_count

        return {
            "white": white_count / total,
            "black": black_count / total
        }

    def predict_next_action_position(self, action):
        """
        Predicts the next position based on the current position and action.
        Returns None if the action would take the robot out of bounds.
        """
        if action == "left":
            next_pos = self.position - 1
            if next_pos < 0:
                return None  # Out of bounds
            return next_pos
        elif action == "right":
            next_pos = self.position + 1
            if next_pos >= WORLD_SIZE:
                return None  # Out of bounds
            return next_pos
        return None  # Should not happen

    def choose_action(self):
        """
        Chooses the action (left/right) that leads to a valid, different position
        where the robot's world model has the highest certainty for the predicted color.
        If no valid moves, it will be stuck (though this scenario should be avoided by design).
        """
        action_certainties = {}
        valid_actions = []

        for action in ACTIONS:
            predicted_next_position = self.predict_next_action_position(action)

            # Only consider actions that lead to a valid, different position
            if predicted_next_position is not None and predicted_next_position != self.position:
                valid_actions.append(action)
                # If the predicted position has never been visited, its probabilities will be 0.5/0.5
                # due to Laplace smoothing. The certainty will be 0.5.
                if predicted_next_position not in self.world_model:
                    # If a position has never been observed, assume 50/50 for certainty
                    action_certainties[action] = 0.5
                else:
                    probs = self.get_color_probabilities_at_position(predicted_next_position)
                    action_certainties[action] = max(probs.values())

        if not valid_actions:
            # This scenario means the robot is at an edge and cannot move to a *different* tile.
            # In a 1D world, if it's at position 0, only 'right' is valid. If at WORLD_SIZE-1, only 'left' is valid.
            # If WORLD_SIZE is 1, it will be stuck. Assuming WORLD_SIZE > 1.
            print("Warning: Robot is stuck! No valid moves to a different tile.")
            # We can't choose an action that forces a move if there are no valid ones.
            # For now, let's just default to 'right' if stuck, or handle it as an error.
            # For this exercise, let's assume WORLD_SIZE > 1, so there's always at least one valid move.
            # If you want it to truly be "stuck" if no valid move, you'd raise an exception or return None.
            # For now, we will return a random action from the original list if no valid ones were found,
            # which will lead to `None` in the next step, essentially showing it couldn't move.
            return ACTIONS[randint(0, len(ACTIONS) - 1)]  # Fallback: will cause an issue in step() if this happens

        # Find the action(s) with the highest certainty among valid actions
        max_certainty = -1
        if action_certainties:  # Ensure action_certainties is not empty
            max_certainty = max(action_certainties.values())
            best_actions = [action for action, certainty in action_certainties.items() if certainty == max_certainty]
            return best_actions[randint(0, len(best_actions) - 1)]
        else:
            # If no valid actions that lead to positions with existing certainty (e.g., all leads to unvisited)
            # This should be covered by the initial `valid_actions` check.
            # If for some reason `action_certainties` is empty but `valid_actions` is not,
            # it implies all valid actions lead to entirely unvisited positions.
            # In that case, pick a random valid action.
            return valid_actions[randint(0, len(valid_actions) - 1)]

    def show_world_model_histograms(self):
        """Visualizes the robot's world model (histograms for each position)."""
        num_positions = WORLD_SIZE
        fig, axs = plt.subplots(1, num_positions, figsize=(num_positions * 3, 4), sharey=True)

        # Handle the case of a single subplot for a small world
        if num_positions == 1:
            axs = [axs]

        for i in range(num_positions):
            position_data = self.world_model[i]
            total_observations = sum(position_data.values())

            if total_observations == 0:
                # If no observations for this position, assume uniform distribution initially for display
                white_prob = 0.5
                black_prob = 0.5
            else:
                white_prob = (position_data["white"] + 1) / (total_observations + 2)  # +2 for Laplace smoothing
                black_prob = (position_data["black"] + 1) / (total_observations + 2)

            colors = TILES
            probabilities = [white_prob, black_prob]

            axs[i].bar(colors, probabilities, color=["lightgray", "dimgray"])
            axs[i].set_title(f'Position: {i}\n({WORLD_TILES[i]} tile)')  # Show actual tile color
            axs[i].set_ylim(0, 1)
            axs[i].set_ylabel('Probability')

        plt.tight_layout()
        plt.suptitle("Robot's World Model: Color Probabilities per Position", y=1.02)
        plt.show()

    def step(self):
        """Performs one step of the robot's simulation."""
        # 1. Robot measures current tile color
        observed_color = self.measure_tile_color()
        print(f"Robot is at position {self.position}, observes {observed_color} tile.")

        # 2. Update world model with the observation
        self.update_world_model(self.position, observed_color)

        # 3. Visualize the current state of the world model
        self.show_world_model_histograms()

        # 4. Robot chooses next action
        action = self.choose_action()

        # 5. Predict the next position. This *must* be a valid, different position now.
        next_position = self.predict_next_action_position(action)

        if next_position is None:
            print(
                f"Robot attempted to move {action} from position {self.position} but it's out of bounds or invalid. Robot stays put.")
            # In this revised logic, choose_action should ideally only return valid moves.
            # If it still returns an invalid one (e.g., due to WORLD_SIZE=1 edge case),
            # the robot will not move.
        else:
            print(f"Robot chooses action: {action}")
            # 6. Robot moves
            self.position = next_position
            self.current_tile_color = get_tile_color(self.position)  # Update current tile color after move
            print(f"Robot moved to position {self.position} (a {self.current_tile_color} tile).")
        print(line_break)


# Initiate Robot
robot = Robot()

# Run robot for n steps
for step_num in range(n_steps):
    print(f"--- Step {step_num + 1}/{n_steps} ---")
    robot.step()