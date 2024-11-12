# #1. Agent Initialization: The agent is initialized with a set of actions and a Q-table to store learned values.
# #2. Learning Parameters: Configured for exploration (to try new actions) and
# #   learning rate (how quickly the agent updates its values).
# #3. Action Selection: Balances between exploring random actions and exploiting the best known action,
# #   depending on the agent’s configuration.
# #4. State and Reward Management: Updates its Q-table based on received rewards and observed outcomes,
# #   which helps it gradually improve its actions.
# #5. Episode Control: Each episode runs until a termination condition (flag found or failure) or
# #   until reaching the maximum steps.
# #6. The agent learns by running multiple episodes, updating its Q-table with each action-response pair,
# #   ultimately optimizing its ability to detect malicious or benign actions.

# import numpy as np
# import sys


# class Agent:
#     # Initializes the agent with actions, verbosity, and learning parameters
#     def __init__(self, actions, verbose=True):
#         self.actions = actions  # List of possible actions the agent can take
#         self.num_actions = len(actions)  # Number of actions
#         self.Q = {
#             (): np.ones(self.num_actions)
#         }  # Q-table initialized for an empty state
#         self.verbose = verbose  # If True, agent prints debug information
#         self.set_learning_options()  # Sets default learning options
#         self.used_actions = []  # Tracks actions taken in the current episode
#         self.steps = 0  # Tracks the number of steps in the current episode
#         self.rewards = 0  # Tracks cumulative reward for the current episode
#         self.total_trials = 0  # Counts the total episodes run
#         self.total_successes = 0  # Counts successful episodes (episodes where termination condition is met)

#     # Sets learning parameters for the agent, allowing exploration, learning rate, discount factor, and max steps
#     def set_learning_options(
#         self, exploration=0.2, learningrate=0.1, discount=0.9, max_step=100
#     ):
#         self.expl = exploration  # Probability of exploring a random action instead of exploiting the best known action
#         self.lr = learningrate  # Learning rate for Q-learning updates
#         self.discount = (
#             discount  # Discount factor for future rewards in Q-learning updates
#         )
#         self.max_step = (
#             max_step  # Maximum steps allowed per episode to prevent infinite loops
#         )

#     # Selects an action for the agent based on exploration-exploitation strategy
#     def _select_action(self, learning=True):
#         if np.random.random() < self.expl and learning:
#             # Chooses a random action with probability `expl` to explore
#             return np.random.randint(0, self.num_actions)
#         else:
#             # Chooses the best known action (highest Q-value) for the current state
#             return np.argmax(self.Q[self.state])

#     # Executes a step in the environment, selecting an action, and receiving feedback
#     def step(self, deterministic=False):
#         self.steps += 1  # Increments step count for current episode
#         action = self._select_action(
#             learning=not deterministic
#         )  # Selects an action (random if exploring, best Q if not)
#         self.used_actions.append(action)  # Tracks the action taken
#         state_resp, reward, termination, debug_msg = self.env.step(
#             action
#         )  # Executes action in environment and receives feedback
#         self.rewards += (
#             reward  # Adds received reward to cumulative rewards for the episode
#         )
#         self._analyze_response(
#             action, state_resp, reward, learning=not deterministic
#         )  # Analyzes the response to the action
#         self.terminated = termination  # Updates whether the episode should terminate based on environment feedback
#         if self.verbose:
#             print(debug_msg)  # Prints debug message if verbosity is enabled

#     # Updates the state after taking an action based on response interpretation (success or failure)
#     def _update_state(self, action_nr, response_interpretation):
#         action_nr += 1  # Adjusts action number (avoids 0-index for interpretation)
#         x = list(
#             set(list(self.state) + [response_interpretation * action_nr])
#         )  # Updates state representation with action outcome
#         x.sort()  # Sorts state representation for consistent ordering
#         self.oldstate = self.state  # Stores current state as `oldstate` before updating
#         self.state = tuple(x)  # Sets the new state representation as a tuple
#         self.Q[self.state] = self.Q.get(
#             self.state, np.ones(self.num_actions)
#         )  # Adds new state to Q-table if not present

#     # Analyzes environment's response to the action, updating Q-value based on response and reward
#     def _analyze_response(self, action, response, reward, learning=True):
#         if response in [1, 4]:  # Response indicates a successful action
#             self._update_state(
#                 action, response_interpretation=1
#             )  # Updates state positively
#             if learning:
#                 self._update_Q(action, reward)  # Updates Q-table if in learning mode
#         elif response in [2, 0, -1]:  # Response indicates an unsuccessful action
#             self._update_state(
#                 action, response_interpretation=-1
#             )  # Updates state negatively
#             if learning:
#                 self._update_Q(action, reward)  # Updates Q-table if in learning mode
#         elif (
#             response == 3
#         ):  # Response indicates a successful terminal action (flag found)
#             self._update_state(
#                 action, response_interpretation=1
#             )  # Updates state positively
#             if learning:
#                 self._update_Q(action, reward)  # Updates Q-table if in learning mode
#         else:
#             print(
#                 "ILLEGAL RESPONSE"
#             )  # If response is unknown, prints an error and exits
#             sys.exit()

#     # Q-learning update rule to adjust the Q-value based on observed reward and future state
#     def _update_Q(self, action, reward):
#         best_action_newstate = np.argmax(
#             self.Q[self.state]
#         )  # Finds the best action for the new state
#         # Q-learning update rule to adjust Q-value for the previous state-action pair
#         self.Q[self.oldstate][action] += self.lr * (
#             reward
#             + self.discount * self.Q[self.state][best_action_newstate]
#             - self.Q[self.oldstate][action]
#         )

#     # Resets agent's state and variables to begin a new episode
#     def reset(self, env):
#         self.env = env  # Assigns the environment to the agent
#         self.terminated = False  # Resets termination flag
#         self.state = ()  # Sets state to an empty tuple
#         self.oldstate = None  # No previous state at the start of the episode
#         self.used_actions = []  # Clears actions taken
#         self.steps = 0  # Resets step counter
#         self.rewards = 0  # Resets reward counter

#     # Runs a full episode, stepping through until the termination condition is met or max steps is reached
#     def run_episode(self, deterministic=False):
#         _, _, self.terminated, _ = (
#             self.env.reset()
#         )  # Resets environment, receiving initial state and termination flag
#         while (
#             not self.terminated and self.steps < self.max_step
#         ):  # Loops until termination or max steps
#             self.step(deterministic=deterministic)  # Takes steps in the environment
#         self.total_trials += 1  # Increments trial counter
#         if self.terminated:
#             self.total_successes += (
#                 1  # If episode ended successfully, increments success counter
#             )
#         return self.terminated  # Returns whether the episode was successful


# agent.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import random
from collections import deque


class QNetwork(tf.keras.Model):
    def __init__(self, action_space):
        super(QNetwork, self).__init__()
        self.dense1 = layers.Dense(64, activation="relu")
        self.dense2 = layers.Dense(64, activation="relu")
        self.output_layer = layers.Dense(action_space, activation="linear")

    # def call(self, state):
    #     x = self.dense1(state)
    #     x = self.dense2(x)
    #     return self.output_layer(x)
    def call(self, state):
        # Expand the state dimensions to match the expected input shape of (batch_size, num_features)
        if len(state.shape) == 1:  # If state is 1D, expand dimensions to make it 2D
            state = tf.expand_dims(
                state, axis=0
            )  # Expand to (1, num_features) if batch size is 1

        x = self.dense1(state)  # Pass state through the first dense layer
        x = self.dense2(x)  # Pass through the second dense layer
        return self.output_layer(x)  # Output layer


class DQNAgent:
    def __init__(
        self,
        action_space,
        state_size,
        alpha=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
    ):
        self.action_space = action_space
        self.state_size = state_size
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha  # Learning rate

        # Create Q-network and target Q-network
        self.q_network = QNetwork(action_space)
        self.target_q_network = QNetwork(action_space)
        self.target_q_network.set_weights(
            self.q_network.get_weights()
        )  # Copy weights to target network

        # Replay buffer for experience replay
        self.replay_buffer = deque(maxlen=2000)

        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.alpha)

    def select_action(self, state):
        # Check if state is a scalar, and if so, expand it to a vector of size self.state_size

        if np.isscalar(state):
            # Convert the scalar state into a vector, for example by creating a one-hot vector
            state = np.zeros(self.state_size)
            state[state] = (
                1  # Assuming state is an index, convert it to one-hot encoding
            )
            state = np.reshape(state, (1, self.state_size))  # Ensure it's a 2D array

        # Ensure state is a 2D array with shape (1, self.state_size)

        if len(state.shape) == 1 and state.size == self.state_size:
            state = np.reshape(state, (1, self.state_size))

        elif state.shape != (1, self.state_size):
            raise ValueError(
                f"Expected state shape (1, {self.state_size}), but got {state.shape}"
            )

        # Exploration or exploitation decision

        if np.random.rand() <= self.epsilon:
            return random.randint(0, self.action_space - 1)  # Explore

        else:
            q_values = self.q_network(state)  # Get Q-values for all actions
            return np.argmax(
                q_values[0]
            )  # Exploit (choose action with highest Q-value)

    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def sample_batch(self, batch_size=32):
        return random.sample(self.replay_buffer, batch_size)

    def train(self, batch_size=32):
        if len(self.replay_buffer) < batch_size:
            return  # Not enough samples to train

        # Sample a batch of experiences
        batch = self.sample_batch(batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to numpy arrays
        states = np.array(states)
        next_states = np.array(next_states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)

        # Get target Q-values from target network
        target_q_values = self.target_q_network(next_states)
        max_next_q_values = np.max(target_q_values, axis=1)
        targets = rewards + (self.gamma * max_next_q_values * (1 - dones))

        # Get current Q-values from the Q-network
        with tf.GradientTape() as tape:
            q_values = self.q_network(states)
            actions_one_hot = tf.one_hot(actions, self.action_space)
            q_values_for_actions = tf.reduce_sum(q_values * actions_one_hot, axis=1)

            # Loss function (mean squared error between target and predicted Q-values)
            loss = tf.reduce_mean((targets - q_values_for_actions) ** 2)

        # Compute gradients and update weights
        grads = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))

        # Update epsilon (decay exploration rate)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_q_network.set_weights(
            self.q_network.get_weights()
        )  # Copy weights to target network


class Agent:
    def __init__(
        self,
        actions,
        alpha=0.5,
        gamma=0.9,
        epsilon=0.8,
        epsilon_decay=0.995,
        min_epsilon=0.1,
    ):
        self.actions = actions
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration-exploitation trade-off
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table = None  # To be initialized during training

    def initialize_q_table(self, num_queries):
        self.q_table = np.zeros((num_queries, len(self.actions)))  # State x Actions

    def select_action(self, state):
        # Exploration or exploitation decision
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, len(self.actions) - 1)  # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit

    def update_q_table(self, state, action, reward, next_state):
        self.q_table[state, action] += self.alpha * (
            reward
            + self.gamma * np.max(self.q_table[next_state])
            - self.q_table[state, action]
        )

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
