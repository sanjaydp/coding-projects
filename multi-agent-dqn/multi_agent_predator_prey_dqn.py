import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
import time
import tensorflow.keras.backend as K
from collections import deque

# Define the PPO model
class PPO(tf.keras.Model):
    def __init__(self, state_size, action_size):
        super(PPO, self).__init__()
        self.dense1 = layers.Dense(64, activation='relu')
        self.dense2 = layers.Dense(64, activation='relu')
        self.output_layer = layers.Dense(action_size, activation='softmax')

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        return self.output_layer(x)

class MultiAgentEnvironment:
    def __init__(self, grid_size, num_predators, num_prey):
        self.grid_size = grid_size
        self.num_predators = num_predators
        self.num_prey = num_prey

        self.predators = [[random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)] for _ in range(num_predators)]
        self.prey = [[random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)] for _ in range(num_prey)]

        self.models = [PPO(state_size=2, action_size=4) for _ in range(num_predators)]
        self.memory = [deque(maxlen=1000) for _ in range(num_predators)]

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.gamma = 0.99
        self.epsilon = 0.1
        self.steps = 0
        self.heatmap = np.zeros((grid_size, grid_size))

        self.fig, self.ax = plt.subplots()
        plt.ion()

    def get_new_position(self, agent, action):
        new_position = agent[:]
        if action == 0 and agent[1] > 0:
            new_position[1] -= 1
        elif action == 1 and agent[1] < self.grid_size - 1:
            new_position[1] += 1
        elif action == 2 and agent[0] > 0:
            new_position[0] -= 1
        elif action == 3 and agent[0] < self.grid_size - 1:
            new_position[0] += 1
        return new_position

    def get_state(self, agent):
        return np.array(agent) / self.grid_size

    def reward(self, predator):
        reward = -0.01
        for prey in self.prey:
            if predator == prey:
                reward += 10
                self.prey.remove(prey)
        return reward

    def select_action(self, model, state):
        state = np.reshape(state, (1, 2))
        prob = model(state)[0].numpy()
        action = np.random.choice(4, p=prob)
        return action

    def update_ppo(self, model, memory):
        states, actions, rewards, next_states = zip(*memory)

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)

        advantages = rewards + self.gamma * np.amax(model(next_states), axis=1) - np.amax(model(states), axis=1)

        with tf.GradientTape() as tape:
            prob = model(states)
            action_prob = tf.reduce_sum(prob * tf.one_hot(actions, 4), axis=1)
            old_prob = K.stop_gradient(action_prob)
            ratio = action_prob / (old_prob + 1e-10)
            clip = tf.clip_by_value(ratio, 1.0 - 0.2, 1.0 + 0.2)
            loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clip * advantages))

        gradients = tape.gradient(loss, model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    def move_predators(self):
        for i, predator in enumerate(self.predators):
            state = self.get_state(predator)
            action = self.select_action(self.models[i], state)
            new_position = self.get_new_position(predator, action)
            reward = self.reward(new_position)
            next_state = self.get_state(new_position)

            self.memory[i].append((state, action, reward, next_state))

            predator[0], predator[1] = new_position[0], new_position[1]
            self.heatmap[new_position[1], new_position[0]] += 1

            if len(self.memory[i]) >= 32:
                self.update_ppo(self.models[i], self.memory[i])
                self.memory[i].clear()

    def move_prey(self):
        for prey in self.prey:
            action = random.choice(range(4))
            new_position = self.get_new_position(prey, action)
            prey[0], prey[1] = new_position[0], new_position[1]

    def render_heatmap(self):
        plt.figure(figsize=(6, 6))
        sns.heatmap(self.heatmap, cmap='coolwarm', cbar=True)
        plt.title('Predator Movement Heatmap')
        plt.show()

    def render(self):
        grid = np.zeros((self.grid_size, self.grid_size))
        for predator in self.predators:
            grid[predator[1], predator[0]] = 1
        for prey in self.prey:
            grid[prey[1], prey[0]] = 2
        self.ax.clear()
        self.ax.matshow(grid, cmap="coolwarm")
        plt.draw()
        plt.pause(0.1)

    def train(self, episodes=500):
        for episode in range(episodes):
            self.predators = [[random.randint(0, self.grid_size - 1), random.randint(0, grid_size - 1)] for _ in range(self.num_predators)]
            self.prey = [[random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)] for _ in range(self.num_prey)]
            steps = 0
            while self.prey and steps < 100:
                self.move_predators()
                self.move_prey()
                self.render()
                steps += 1
            print(f"Episode {episode + 1}/{episodes} - Steps: {steps}")

            if (episode + 1) % 10 == 0:
                self.render_heatmap()

# Run the environment
env = MultiAgentEnvironment(grid_size=10, num_predators=3, num_prey=2)
env.train(episodes=50)
