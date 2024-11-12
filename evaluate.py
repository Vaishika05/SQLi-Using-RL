import numpy as np


# Evaluates a trained model's performance in the environment over a specified number of steps.
def evaluate_agent(agent, env, num_episodes=10, deterministic=False):
    """
    Evaluates the agent over a specified number of episodes.

    Parameters:
    - agent: Instance of the Agent class.
    - env: Environment in which the agent operates.
    - num_episodes: Number of episodes to run for evaluation.
    - deterministic: If True, the agent will not explore (only exploit known Q-values).

    Returns:
    - mean_reward: Average reward obtained across all episodes.
    - success_rate: Proportion of successful episodes.
    """
    total_rewards = 0
    successful_episodes = 0

    for episode in range(num_episodes):
        agent.reset(env)  # Reset the agent and environment for each episode
        success = agent.run_episode(deterministic=deterministic)  # Run a single episode

        # Accumulate rewards and count successful episodes
        total_rewards += agent.rewards
        if success:
            successful_episodes += 1

    # Calculate mean reward and success rate
    mean_reward = total_rewards / num_episodes
    success_rate = successful_episodes / num_episodes

    print(f"Average Reward: {mean_reward}")
    print(f"Success Rate: {success_rate * 100:.2f}%")

    return mean_reward, success_rate


def evaluate_random(env, num_steps=1000):
    episode_rewards = [0.0]  # List to store rewards per episode
    obs = env.reset()  # Resets the environment and gets the initial observation
    for i in range(num_steps):
        # Randomly select an action from available actions
        action = np.random.choice(env.action_space)
        obs, reward, done, _ = env.step(action)  # Takes the random action
        episode_rewards[
            -1
        ] += reward  # Adds the reward from this step to the current episode's reward

        if done:  # If the environment signals episode completion
            obs = env.reset()  # Resets the environment for the next episode
            episode_rewards.append(0.0)  # Starts a new episode's reward count

    mean_reward = round(
        np.mean(episode_rewards), 3
    )  # Calculates the average reward across episodes
    return (
        mean_reward,
        len(episode_rewards) - 1,
    )  # Returns mean reward and total episodes completed
