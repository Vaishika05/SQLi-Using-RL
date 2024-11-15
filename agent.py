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


# agent.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import random
from collections import deque
import random


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
        """
        Initialize the Agent with its set of actions and learning parameters.

        :param actions: List of possible actions the agent can take.
        :param alpha: Learning rate for Q-learning updates.
        :param gamma: Discount factor for future rewards.
        :param epsilon: Initial exploration rate.
        :param epsilon_decay: Rate at which epsilon decays over episodes.
        :param min_epsilon: Minimum exploration rate.
        """
        self.actions = actions
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration-exploitation trade-off
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table = None  # Initialize Q-table as None

    def initialize_q_table(self, num_queries):
        """
        Initialize the Q-table with zeros. Assumes state space is discrete and represents
        states by indices from 0 to num_queries - 1.

        :param num_queries: The number of possible states (queries) the agent can encounter.
        """
        self.q_table = np.zeros((num_queries, len(self.actions)))  # State x Actions

    def select_action(self, state):
        """
        Select an action based on the current Q-table and epsilon-greedy policy.

        :param state: The current state, represented by an index.
        :return: The selected action index.
        """
        # Use epsilon-greedy approach: Explore or exploit
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(
                0, len(self.actions) - 1
            )  # Explore with a random action
        else:
            # Exploit the best-known action if Q-table is initialized, otherwise choose randomly
            if self.q_table is not None:
                return np.argmax(
                    self.q_table[state]
                )  # Choose best action for the given state
            else:
                return random.randint(0, len(self.actions) - 1)

    def update_q_table(self, state, action, reward, next_state):
        """
        Update the Q-table based on the observed reward and next state.

        :param state: Current state index.
        :param action: Action taken.
        :param reward: Observed reward after taking the action.
        :param next_state: Next state index after the action.
        """
        if self.q_table is not None:
            # Q-learning update rule
            best_future_value = (
                np.max(self.q_table[next_state]) if next_state is not None else 0
            )
            self.q_table[state, action] += self.alpha * (
                reward + self.gamma * best_future_value - self.q_table[state, action]
            )

    def decay_epsilon(self):
        """
        Decay the exploration rate (epsilon) after each episode to favor exploitation over time.
        """
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def reset_epsilon(self, epsilon=None):
        """
        Reset epsilon to its original or a specified value, typically done between training phases.

        :param epsilon: Optional parameter to set epsilon to a specific value.
        """
        if epsilon is not None:
            self.epsilon = epsilon
        else:
            self.epsilon = 0.8  # Default initial value, can be adjusted
