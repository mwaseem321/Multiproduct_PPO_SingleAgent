import torch
from NASH import *
from PPO import PPO
from ProductionLine import Multiproduct

# Main function
def main():
    env = Multiproduct(ptypes, n, b, B, Tp, ng, MTTR, MTBF, T, Tl, Tu)
    input_dim = 15 #11 #29  # len(env.state)
    output_dim = 10  # len(env.action_list)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ppo = PPO(env, input_dim, output_dim, device)
    num_episodes = 2000
    # T = 200  # Time horizon for each episode

    # Train the PPO agent
    ppo.train(num_episodes, T, save_model_path="ppo_model.pth")
    # Evaluate the trained model
    num_evaluation_episodes = 50
    mean_reward, std_reward = ppo.evaluate(num_evaluation_episodes, T, model_path="ppo_model.pth")
    print(f"Evaluation - Mean Reward: {mean_reward}, Std Deviation: {std_reward}")

if __name__ == '__main__':
    main()

