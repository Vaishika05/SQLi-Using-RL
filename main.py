# # main.py

# from mockSQLenv import SQLInjectionEnv  # Now matches the class name
# from agent import Agent  # Ensure this is defined correctly
# from generate_actions import generate_actions  # Ensure actions generator is defined

# if __name__ == "__main__":
#     env = SQLInjectionEnv()
#     actions = generate_actions()  # Generate a list of SQL actions for the agent

#     agent = Agent(actions)
#     agent.reset(env)

#     # Train the agent over multiple episodes
#     num_training_episodes = 1000
#     for episode in range(num_training_episodes):
#         agent.run_episode()

#     print("Training complete.")

#     # Accepting user input for detection
#     user_query = input("Enter a SQL query to detect if it's malicious: ")
#     env.set_query(user_query)  # Assuming `set_query` is defined in SQLInjectionEnv

#     agent.reset(env)
#     agent.run_episode(deterministic=True)

#     if agent.terminated:
#         print("The agent flagged the query as potentially malicious.")
#     else:
#         print("The agent did not flag the query as malicious.")


from joblib import load
import pandas as pd
from mockSQLenv import SQLInjectionEnv  # Ensure this is the correct import path
import numpy as np
from agent import Agent


def main():
    # Load the trained SQLi detection model
    model = load("sqli_model.pkl")

    # Load your dataset (or queries list) for the environment
    dataset = pd.read_csv("sqli_dataset.csv")  # Assuming you have a dataset file
    queries = dataset["query"].tolist()  # List of SQL queries (benign or malicious)

    # Initialize the environment with the model and queries
    env = SQLInjectionEnv(model, queries)

    # Accepting user input for detection
    user_query = input("Enter a SQL query to detect if it's malicious: ")

    # Set the query in the environment
    env.set_query(user_query)  # Assuming `set_query` is defined in SQLInjectionEnv

    # Check if the query is malicious or not
    result = env.check_bypass(
        user_query
    )  # Returns True for benign (bypass) or False for malicious

    # Display the result
    if result:
        print("The query is detected as benign (bypass was successful).")
    else:
        print("The query is detected as malicious.")


if __name__ == "__main__":
    main()
