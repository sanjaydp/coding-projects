import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import time

# Define the DQN model
class DQN(tf.keras.Model):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.dense1 = layers.Dense(64, activation='relu')
        self.dense2 = layers.Dense(64, activation='relu')
        self.output_layer = layers.Dense(action_size)

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

        self.models = [DQN(state_size=2, action_size=4) for _ in range(num_predators)]
        self.target_models = [DQN(state_size=2, action_size=4) for _ in range(num_predators)]

        for model, target_model in zip(self.models, self.target_models):
            target_model.set_weights(model.get_weights())

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.gamma = 0.99
        self.epsilon = 0.1
        self.steps = 0

        self.fig, self.ax = plt.subplots()
        plt.ion()

    def get_new_position(self, agent, action):
        new_position = agent[:]
        if action == 0 and agent[1] > 0: # UP
            new_position[1] -= 1
        elif action == 1 and agent[1] < self.grid_size - 1: # DOWN
            new_position[1] += 1
        elif action == 2 and agent[0] > 0: # LEFT
            new_position[0] -= 1
        elif action == 3 and agent[0] < self.grid_size - 1: # RIGHT
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

    def update_dqn(self, model, target_model, state, action, reward, next_state):
        target = reward + self.gamma * np.max(target_model(next_state[None])[0])
        target_value = model(state[None])[0].numpy()
        target_value[action] = target

        with tf.GradientTape() as tape:
            q_values = model(state[None])
            loss = tf.keras.losses.mean_squared_error(target_value, q_values[0])

        gradients = tape.gradient(loss, model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    def select_action(self, model, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(4))
        else:
            return np.argmax(model(state[None])[0].numpy())

    def share_knowledge(self):
        prey_positions = [tuple(prey) for prey in self.prey]
        for predator in self.predators:
            print(f"Predator at {predator} sees prey at {prey_positions}")

    def move_predators(self):
        for i, predator in enumerate(self.predators):
            state = self.get_state(predator)
            action = self.select_action(self.models[i], state)
            new_position = self.get_new_position(predator, action)
            reward = self.reward(new_position)
            next_state = self.get_state(new_position)

            self.update_dqn(self.models[i], self.target_models[i], state, action, reward, next_state)

            predator[0], predator[1] = new_position[0], new_position[1]

            self.apply_flocking(predator)

    def move_prey(self):
        for prey in self.prey:
            action = random.choice(range(4))
            new_position = self.get_new_position(prey, action)
            prey[0], prey[1] = new_position[0], new_position[1]

    def apply_flocking(self, predator):
        alignment = [0, 0]
        cohesion = [0, 0]
        separation = [0, 0]
        num_neighbors = 0

        for other in self.predators:
            if other != predator:
                dist = self.distance(predator, other)
                if dist < 3:
                    alignment[0] += other[0]
                    alignment[1] += other[1]
                    cohesion[0] += other[0]
                    cohesion[1] += other[1]
                    separation[0] += predator[0] - other[0]
                    separation[1] += predator[1] - other[1]
                    num_neighbors += 1

        if num_neighbors > 0:
            alignment = [x / num_neighbors for x in alignment]
            cohesion = [x / num_neighbors for x in cohesion]
            separation = [x / num_neighbors for x in separation]

            predator[0] += (alignment[0] + cohesion[0] + separation[0]) / 3
            predator[1] += (alignment[1] + cohesion[1] + separation[1]) / 3

    def distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

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
            self.predators = [[random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)] for _ in range(self.num_predators)]
            self.prey = [[random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)] for _ in range(self.num_prey)]
            steps = 0
            while self.prey and steps < 100:
                self.share_knowledge()
                self.move_predators()
                self.move_prey()
                self.render()
                steps += 1
            print(f"Episode {episode + 1}/{episodes} - Steps: {steps}")

# Run the environment
env = MultiAgentEnvironment(grid_size=10, num_predators=3, num_prey=2)
env.train(episodes=50)
