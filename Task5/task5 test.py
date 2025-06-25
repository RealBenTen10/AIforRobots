import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Setup
K = 3  # Number of arms
T = 300  # Rounds
episodes = 10
true_means = np.random.uniform(-1, 1, 3)
true_stds = np.random.uniform(0.2, 1.2, 3)

torch.manual_seed(42)
device = torch.device("cpu")

# Simple feedforward regression model
class RewardPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.model(x)

# Initialize a model and optimizer per arm
models = [RewardPredictor().to(device) for _ in range(K)]
loss_fn = nn.MSELoss()
for i in range(episodes):
    np.random.seed(42)
    lr = 0.0001 - i/1000000
    optimizers = [optim.Adam(m.parameters(), lr=lr) for m in models]
    # Data storage
    errors = []
    chosen_arms = []
    predictions = []
    rewards = []
    observations = [[] for _ in range(K)]

    for t in range(1, T + 1):

        t_tensor = torch.tensor([[t / T]], dtype=torch.float32).to(device)

        # Predict reward for each arm
        preds = [models[i](t_tensor).item() for i in range(K)]
        #print("Preds: ", preds)

        # Choose the arm with highest predicted reward (or lowest predicted error)
        arm = int(np.argmax(preds))
        chosen_arms.append(arm)
        pred = preds[arm]
        predictions.append(pred)

        # Sample reward
        reward = np.random.normal(true_means[arm], true_stds[arm])
        rewards.append(reward)
        observations[arm].append((t / T, reward))

        # Calculate prediction error
        errors.append(abs(reward - pred))

        # Train model on new data point
        x = torch.tensor([[t / T]], dtype=torch.float32).to(device)
        y = torch.tensor([[reward]], dtype=torch.float32).to(device)
        optimizers[arm].zero_grad()
        output = models[arm](x)
        loss = loss_fn(output, y)
        loss.backward()
        optimizers[arm].step()
    print(observations)
    print(f"For the first round the prediction was {predictions[0]} and the reward was {rewards[0]}")
    print(f"For round 10 the prediction was {predictions[9]} and the reward was {rewards[9]}")
    print(f"For round 100 the prediction was {predictions[99]} and the reward was {rewards[99]}")
    print(f"For round 299 the prediction was {predictions[298]} and the reward was {rewards[298]}")
    print(f"Total prediction error for episode {i} and lr {round(lr, 7)}: {sum(errors):.2f}")
    print("Arm selection counts:", np.bincount(np.array(chosen_arms)))
