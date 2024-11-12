# # mockSQLenv.py

# import numpy as np


# class SQLInjectionEnv:
#     def __init__(self):
#         self.action_space = np.arange(10)  # Define action space size if needed

#     def set_query(self, query):
#         """Sets the user query for evaluation and resets environment state if necessary."""
#         self.query = query

#     def step(self, action):
#         response = np.random.choice([1, 2, 3, 4, 0, -1])  # Simulate response
#         reward = 1 if response == 3 else 0
#         terminated = response == 3
#         debug_msg = f"Action: {action}, Response: {response}"
#         return response, reward, terminated, debug_msg

#     def reset(self):
#         return None, 0, False, "Environment reset"


# mockSQLenv.py
# import numpy as np
# import pandas as pd
# import random
# import urllib.parse  # For URL encoding
# from joblib import load
# from agent import DQNAgent

# # Load model and dataset
# model = load("sqli_model.pkl")
# dataset = pd.read_csv("sqli_dataset.csv")
# queries = dataset["query"].tolist()


# # Helper function for hex encoding
# def hex_encode(input_str):
#     return "".join(hex(ord(c))[2:] for c in input_str)


# class SQLInjectionEnv:
#     def __init__(self, model, queries):
#         self.model = model
#         self.queries = queries
#         self.unique_payloads = set()
#         self.actions = [
#             "add OR 1=1",
#             "add comment --",
#             "add UNION",
#             "modify quotes",
#             "base",
#             "add UNION SELECT",
#             "add time-based sleep",
#             "error-based SQLi",
#             "add OR 1=1 --",
#             "add OR 'a'='a'",
#             "add SLEEP(5)",
#             "add IF(1=1, SLEEP(5), 0) --",
#             "hex encode OR 1=1",
#             "URL encode UNION SELECT",
#             "nested subquery SELECT FROM information_schema.tables",
#         ]

#     def set_query(self, query):
#         self.query = query

#     def check_bypass(self, query):
#         prediction = self.model.predict([query])[0]
#         return prediction == 0  # 0 indicates a normal query (bypass success)

#     def get_reward(self, query):
#         reward = 1 if self.check_bypass(query) else -1
#         if query not in self.unique_payloads:
#             reward += 0.1  # Diversity bonus for unique payloads
#             self.unique_payloads.add(query)
#         return reward

#     def modify_query(self, query, action):
#         # Dictionary of action-based query modifications
#         modifications = {
#             0: query + " OR 1=1",
#             1: query + " --",
#             2: query + " UNION SELECT *",
#             3: query.replace("'", '"'),
#             4: query,
#             5: query + " UNION SELECT * FROM information_schema.tables",
#             6: query + " AND SLEEP(5)",
#             7: query + " AND 1=CONVERT(int, (SELECT @@version))",
#             8: query + " OR 1=1 --",
#             9: query + " OR 'a'='a'",
#             10: query + " AND SLEEP(5)",
#             11: query + " AND IF(1=1, SLEEP(5), 0) --",
#             12: query + " OR " + hex_encode("1=1"),  # Hex encoding example
#             13: query
#             + " "
#             + urllib.parse.quote("UNION SELECT *"),  # URL encode UNION SELECT
#             14: query
#             + " UNION SELECT column_name FROM (SELECT * FROM information_schema.tables) AS sub",  # Nested subquery
#         }
#         return modifications.get(action, query)


# # Initialize environment and agent
# # env = SQLInjectionEnv(model, queries)
# # agent = Agent(actions=env.actions)
# # agent.initialize_q_table(len(queries))

# # # Training loop
# # for episode in range(50):  # Number of episodes
# #     state = random.randint(0, len(queries) - 1)
# #     query = queries[state]
# #     done = False
# #     steps = 0

# #     while not done and steps < 200:  # Max steps
# #         action = agent.select_action(state)
# #         modified_query = env.modify_query(query, action)

# #         reward = env.get_reward(modified_query)

# #         next_state = random.randint(0, len(queries) - 1)  # Move to next state

# #         agent.update_q_table(state, action, reward, next_state)

# #         steps += 1
# #         if reward > 0:
# #             done = True

# #         state = next_state

# #     agent.decay_epsilon()
# #     print(f"Episode {episode + 1} finished after {steps} steps.")

# # # Save the Q-table
# # np.save("q_table.npy", agent.q_table)


# # Initialize environment and agent
# env = SQLInjectionEnv(model, queries)
# action_space = len(env.actions)
# state_size = len(env.queries)  # or a feature vector size for each query
# agent = DQNAgent(action_space=action_space, state_size=state_size)

# # Training loop
# for episode in range(50):  # Number of episodes
#     state = random.randint(0, len(queries) - 1)  # Random initial state (query index)
#     query = queries[state]
#     done = False
#     steps = 0

#     while not done and steps < 200:  # Max steps per episode
#         action = agent.select_action(
#             state
#         )  # Select action based on epsilon-greedy policy
#         modified_query = env.modify_query(query, action)  # Modify query based on action

#         reward = env.get_reward(modified_query)  # Get reward from the environment

#         next_state = random.randint(
#             0, len(queries) - 1
#         )  # Move to next state (random for now)

#         # Store experience in replay buffer
#         agent.store_experience(state, action, reward, next_state, done)

#         # Train the model
#         agent.train(batch_size=32)

#         # Update state
#         state = next_state
#         steps += 1

#         if reward > 0:  # Terminate if the agent achieves success
#             done = True

#     # Update target network periodically
#     if episode % 10 == 0:
#         agent.update_target_network()

#     print(f"Episode {episode + 1} finished after {steps} steps.")

# # Save the Q-network model
# agent.q_network.save("dqn_model.h5")


import numpy as np
import pandas as pd
import random
import urllib.parse  # For URL encoding
from joblib import load


# Helper function for hex encoding
def hex_encode(input_str):
    return "".join(hex(ord(c))[2:] for c in input_str)


class SQLInjectionEnv:
    def __init__(self, model, queries):
        self.model = model
        self.queries = queries
        self.state_size = 30607  # Define the state size (length of the state vector)
        self.unique_payloads = set()
        self.actions = [
            "add OR 1=1",
            "add comment --",
            "add UNION",
            "modify quotes",
            "base",
            "add UNION SELECT",
            "add time-based sleep",
            "error-based SQLi",
            "add OR 1=1 --",
            "add OR 'a'='a'",
            "add SLEEP(5)",
            "add IF(1=1, SLEEP(5), 0) --",
            "hex encode OR 1=1",
            "URL encode UNION SELECT",
            "nested subquery SELECT FROM information_schema.tables",
        ]

    def set_query(self, query):
        """Sets the user query for evaluation and resets environment state if necessary."""
        self.query = query

    def check_bypass(self, query):
        """Check if the query passes the security check"""
        prediction = self.model.predict([query])[0]
        return prediction == 0  # 0 indicates a normal query (bypass success)

    def get_reward(self, query):
        """Return reward for a given query"""
        reward = 1 if self.check_bypass(query) else -1
        if query not in self.unique_payloads:
            reward += 0.1  # Diversity bonus for unique payloads
            self.unique_payloads.add(query)
        return reward

    def modify_query(self, query, action):
        """Modify the query based on the selected action"""
        modifications = {
            0: query + " OR 1=1",
            1: query + " --",
            2: query + " UNION SELECT *",
            3: query.replace("'", '"'),
            4: query,
            5: query + " UNION SELECT * FROM information_schema.tables",
            6: query + " AND SLEEP(5)",
            7: query + " AND 1=CONVERT(int, (SELECT @@version))",
            8: query + " OR 1=1 --",
            9: query + " OR 'a'='a'",
            10: query + " AND SLEEP(5)",
            11: query + " AND IF(1=1, SLEEP(5), 0) --",
            12: query + " OR " + hex_encode("1=1"),
            13: query + " " + urllib.parse.quote("UNION SELECT *"),
            14: query
            + " UNION SELECT column_name FROM (SELECT * FROM information_schema.tables) AS sub",
        }
        return modifications.get(action, query)

    def reset(self):
        """Reset the environment and return the initial state"""
        state = random.randint(
            0, len(self.queries) - 1
        )  # Randomly select a query index
        state_vector = np.zeros(
            self.state_size
        )  # Create a zero vector of the required state size
        state_vector[state] = 1  # Set the appropriate index to 1 (one-hot encoding)
        return state_vector  # Return the state as a vector

    def step(self, action):
        """Simulate the action in the environment and return the next state and reward"""
        response = np.random.choice([1, 2, 3, 4, 0, -1])  # Simulate response
        reward = 1 if response == 3 else 0
        terminated = response == 3  # If response is 3, we consider the episode as done
        debug_msg = f"Action: {action}, Response: {response}"

        # Get the next state by randomly selecting a query index
        next_state = random.randint(0, len(self.queries) - 1)
        next_state_vector = np.zeros(self.state_size)
        next_state_vector[next_state] = 1  # One-hot encoding for the next state

        return next_state_vector, reward, terminated, debug_msg
