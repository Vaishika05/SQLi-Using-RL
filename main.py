from joblib import load
import pandas as pd
import numpy as np
from agent import DQNAgent
from mockSQLenv import SQLInjectionEnv


def main():
    # Load the trained SQLi detection model
    model = load("sqli_model.pkl")

    # Load your dataset for the environment
    dataset = pd.read_csv("sqli_dataset.csv")
    queries = dataset["query"].tolist()

    # Initialize SQL Injection Environment
    env = SQLInjectionEnv(model, queries)

    # Define parameters for DQNAgent
    action_space = len(env.actions)
    state_size = env.state_size

    # Initialize DQNAgent
    agent = DQNAgent(action_space=action_space, state_size=state_size)

    # Training parameters
    episodes = 10  # Number of training episodes
    max_steps = 50  # Maximum steps per episode to prevent infinite loops

    for episode in range(episodes):
        # Reset environment and get the initial state
        state = env.reset()

        done = False
        total_reward = 0
        step = 0  # Step counter for the episode

        while not done and step < max_steps:
            # Select an action using the DQN agent
            action = agent.select_action(state)

            # Apply the selected action in the environment
            next_state, reward, done, debug_msg = env.step(action)

            # Print debugging info for each step
            print(debug_msg)

            # Store experience in replay buffer
            agent.store_experience(state, action, reward, next_state, done)

            # Train the agent after storing experience
            agent.train(batch_size=32)

            # Update state and total reward
            state = next_state
            total_reward += reward
            step += 1  # Increment step counter

        print(
            f"Episode {episode + 1}/{episodes} - Total Reward: {total_reward}, Steps: {step}"
        )

        # Update target network periodically (e.g., every 10 episodes)
        if episode % 10 == 0:
            agent.update_target_network()

    print("Training complete.")

    # Accepting user input for detection
    user_query = input("Enter a SQL query to detect if it's malicious: ")
    env.set_query(user_query)  # Assuming `set_query` is defined in SQLInjectionEnv

    # Use environment's check_bypass to detect malicious intent
    result = env.check_bypass(user_query)
    if result:
        print("The query is detected as benign (bypass was successful).")
    else:
        print("The query is detected as malicious.")


if __name__ == "__main__":
    main()
