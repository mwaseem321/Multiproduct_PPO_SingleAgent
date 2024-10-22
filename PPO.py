import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import pickle

from NASH import *
from ProductionLine import Multiproduct


# Actor-Critic Model
class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.actor = nn.Linear(256, output_dim)
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x

    def get_action(self, x):
        x = self.forward(x)
        logits = self.actor(x)
        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)

        # Sample two independent actions, one for each robot
        action1 = dist.sample()  # Sample action for robot 1
        action2 = dist.sample()  # Sample action for robot 2

        # Combine the two actions into a single tensor
        actions = torch.stack([action1, action2])

        # Log probabilities of the selected actions
        log_probs = torch.stack([dist.log_prob(action1), dist.log_prob(action2)])

        return actions, log_probs

    def evaluate(self, x, action):
        x = self.forward(x)
        logits = self.actor(x)
        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.critic(x)
        return log_prob, value, entropy


# PPO Algorithm
class PPO:
    def __init__(self, env, input_dim, output_dim, device='cpu'):
        self.env = env
        self.device = device
        self.model = ActorCritic(input_dim, output_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.gamma = 0.99
        self.eps_clip = 0.1
        self.K_epochs = 4
        self.env = env

    def compute_advantages(self, rewards, values, next_value, dones):
        advantages = []
        gae = 0
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * next_value * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * gae * (1 - dones[step])
            advantages.insert(0, gae)
            next_value = values[step]
        return advantages

    def update(self, states, actions, log_probs, returns, advantages):
        for _ in range(self.K_epochs):
            new_log_probs, values, entropy = self.model.evaluate(states, actions)
            ratios = torch.exp(new_log_probs - log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(returns, values)
            loss = actor_loss + 0.5 * critic_loss - 0.02 * entropy.mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        state = (state - state.mean()) / (state.std() + 1e-8)
        action, log_prob = self.model.get_action(state)
        return action.cpu().numpy(), log_prob.cpu().detach()

    def train(self, num_episodes, T, save_model_path="ppo_model.pth"):
        all_rewards = []
        state_action_reward_data = []

        for episode in range(num_episodes):
            print("Training Episode: ", episode)
            state = self.env.reset()
            done = False
            episode_reward = 0
            episode_data = []  # Collect state, action, reward for each step

            for t in range(T):
                action, log_prob = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward

                # Store the state, action, reward
                episode_data.append((state, action, reward))

                state = next_state
                if done:
                    break

            all_rewards.append(episode_reward)
            state_action_reward_data.append(episode_data)
            print(f"Training Episode {episode + 1}/{num_episodes} Reward: {episode_reward}")

        # Save the model after training
        torch.save(self.model.state_dict(), save_model_path)
        print(f"Model saved to {save_model_path}")

        # Plot the rewards at the end of training
        plt.figure(figsize=(10, 5))
        plt.plot(all_rewards, label='Episode Reward')
        plt.xlabel('Episodes')
        plt.ylabel('Total Reward')
        plt.title('Total Reward per Episode During Training')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_rewards_plot.png')  # Save the plot to a file
        plt.show()  # Display the plot

        return all_rewards

    def evaluate(self, num_evaluation_episodes, T, model_path="ppo_model.pth"):
        # Load the trained model for evaluation
        self.model.load_state_dict(torch.load(model_path))
        print(f"Model loaded from {model_path}")

        all_rewards = []
        state_action_reward_data = []

        for episode in range(num_evaluation_episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0
            episode_data = []  # To store state, action, reward for each step in this episode

            for t in range(T):
                action, _ = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)

                # Save state, action, reward for this timestep
                episode_data.append((state, action, reward))

                episode_reward += reward
                state = next_state
                if done:
                    break

            # Save episode data (state, action, reward for each timestep in this episode)
            state_action_reward_data.append(episode_data)
            all_rewards.append(episode_reward)

        # Save the state, action, reward data to a text file
        with open("evaluation_state_action_reward_data.txt", "w") as f:
            for episode_num, episode_data in enumerate(state_action_reward_data):
                f.write(f"Episode {episode_num + 1}:\n")
                for step_data in episode_data:
                    state, action, reward = step_data
                    f.write(f"State: {state}, Action: {action}, Reward: {reward}\n")
                f.write("\n")

        print("State, Action, Reward data saved to 'evaluation_state_action_reward_data.txt'")

        mean_reward = np.mean(all_rewards)
        std_reward = np.std(all_rewards)

        # Plot the rewards at the end of evaluation
        plt.figure(figsize=(10, 5))
        plt.plot(all_rewards, label='Episode Reward')
        plt.xlabel('Evaluation Episodes')
        plt.ylabel('Total Reward')
        plt.title('Total Reward per Episode During Evaluation')
        plt.legend()
        plt.grid(True)
        plt.savefig('evaluation_rewards_plot.png')  # Save the plot to a file
        plt.show()  # Display the plot

        return mean_reward, std_reward



