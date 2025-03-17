import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
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

        self.prey_q_table = np.zeros((grid_size, grid_size, 4))

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

    def select_prey_action(self, prey):
        state = tuple(prey)
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(4))
        else:
            return np.argmax(self.prey_q_table[state[0], state[1]])

    def update_prey_q_table(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.prey_q_table[next_state[0], next_state[1]])
        target = reward + self.gamma * self.prey_q_table[next_state[0], next_state[1], best_next_action]
        self.prey_q_table[state[0], state[1], action] += self.epsilon * (target - self.prey_q_table[state[0], state[1], action])

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
            self.heatmap[new_position[1], new_position[0]] += 1

    def move_prey(self):
        for prey in self.prey:
            state = tuple(prey)
            action = self.select_prey_action(prey)
            new_position = self.get_new_position(prey, action)
            prey[0], prey[1] = new_position[0], new_position[1]

            reward = -0.1
            for predator in self.predators:
                if predator == prey:
                    reward += -10
            next_state = tuple(new_position)
            self.update_prey_q_table(state, action, reward, next_state)

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

            if (episode + 1) % 10 == 0:
                self.render_heatmap()

# Run the environment
env = MultiAgentEnvironment(grid_size=10, num_predators=3, num_prey=2)
env.train(episodes=50)
