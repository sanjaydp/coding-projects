import random
import numpy as np
import matplotlib.pyplot as plt
import time

class PredatorPreyQLearning:
    def __init__(self, grid_size, num_predators, num_prey, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1):
        self.grid_size = grid_size
        self.num_predators = num_predators
        self.num_prey = num_prey
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate

        # Initialize positions of predators and prey
        self.predators = [[random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)] for _ in range(num_predators)]
        self.prey = [[random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)] for _ in range(num_prey)]

        # Q-tables for predators
        self.q_tables = [np.zeros((grid_size, grid_size, 4)) for _ in range(num_predators)]

        # Actions: 0 = UP, 1 = DOWN, 2 = LEFT, 3 = RIGHT
        self.actions = ["UP", "DOWN", "LEFT", "RIGHT"]

        self.fig, self.ax = plt.subplots()
        plt.ion()

    def get_new_position(self, agent, action):
        new_position = agent[:]
        if action == 0 and agent[1] > 0:  # UP
            new_position[1] -= 1
        elif action == 1 and agent[1] < self.grid_size - 1:  # DOWN
            new_position[1] += 1
        elif action == 2 and agent[0] > 0:  # LEFT
            new_position[0] -= 1
        elif action == 3 and agent[0] < self.grid_size - 1:  # RIGHT
            new_position[0] += 1
        return new_position

    def distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def select_action(self, predator_index):
        if random.uniform(0, 1) < self.exploration_rate:
            return random.choice(range(4))  # Explore (random action)
        state = tuple(self.predators[predator_index])
        return np.argmax(self.q_tables[predator_index][state[0], state[1]])  # Exploit (best known action)

    def reward(self, predator_index):
        reward = 0
        for prey_pos in self.prey:
            if self.predators[predator_index] == prey_pos:
                reward += 10  # Positive reward for capture
            else:
                reward -= 1  # Small penalty for not capturing
        return reward

    def update_q_table(self, predator_index, old_state, action, reward, new_state):
        q_table = self.q_tables[predator_index]
        old_q_value = q_table[old_state[0], old_state[1], action]
        best_future_q = np.max(q_table[new_state[0], new_state[1]])

        # Q-learning update equation
        new_q_value = old_q_value + self.learning_rate * (reward + self.discount_factor * best_future_q - old_q_value)
        q_table[old_state[0], old_state[1], action] = new_q_value

    def step(self):
        for i in range(self.num_predators):
            old_state = tuple(self.predators[i])
            action = self.select_action(i)
            new_position = self.get_new_position(self.predators[i], action)
            self.predators[i] = new_position
            reward = self.reward(i)
            new_state = tuple(new_position)
            self.update_q_table(i, old_state, action, reward, new_state)

        # Move prey randomly
        for prey in self.prey:
            action = random.choice(range(4))
            prey[:] = self.get_new_position(prey, action)

        # Check for capture
        for predator in self.predators:
            for prey in self.prey:
                if predator == prey:
                    self.prey.remove(prey)

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

    def train(self, episodes=1000):
        for episode in range(episodes):
            self.predators = [[random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)] for _ in range(self.num_predators)]
            self.prey = [[random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)] for _ in range(self.num_prey)]
            steps = 0
            while self.prey and steps < 100:
                self.step()
                self.render()
                steps += 1
            print(f"Episode {episode + 1}/{episodes} – Steps: {steps}")

        plt.ioff()
        plt.show()

# Environment settings
env = PredatorPreyQLearning(grid_size=10, num_predators=3, num_prey=2)
env.train(episodes=50)
