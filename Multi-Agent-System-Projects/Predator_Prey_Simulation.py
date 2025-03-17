import random
import time
import matplotlib.pyplot as plt
import numpy as np

class PredatorPreyEnvironment:
    def __init__(self, grid_size, num_predators, num_prey):
        self.grid_size = grid_size
        self.predators = [[random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)] for _ in range(num_predators)]
        self.prey = [[random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)] for _ in range(num_prey)]
        self.steps = 0
        self.fig, self.ax = plt.subplots()
        plt.ion()

    def get_new_position(self, agent, move):
        new_position = agent[:]
        if move == "UP" and agent[1] < self.grid_size - 1:
            new_position[1] += 1
        elif move == "DOWN" and agent[1] > 0:
            new_position[1] -= 1
        elif move == "LEFT" and agent[0] > 0:
            new_position[0] -= 1
        elif move == "RIGHT" and agent[0] < self.grid_size - 1:
            new_position[0] += 1
        return new_position
    
    def distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def is_collision(self, pos1, pos2):
        return pos1 == pos2

    def move_predator(self, predator):
        # Find the closest prey
        target_prey = min(self.prey, key=lambda p: self.distance(predator, p))
        
        # Move toward the closest prey using Manhattan distance
        move_options = ["UP", "DOWN", "LEFT", "RIGHT"]
        best_move = None
        shortest_distance = float('inf')

        for move in move_options:
            new_position = self.get_new_position(predator, move)
            dist = self.distance(new_position, target_prey)
            if dist < shortest_distance:
                best_move = move
                shortest_distance = dist

        if best_move:
            new_position = self.get_new_position(predator, best_move)
            predator[0], predator[1] = new_position[0], new_position[1]

    def move_prey(self, prey):
        move_options = ["UP", "DOWN", "LEFT", "RIGHT"]
        random.shuffle(move_options)

        best_move = None
        longest_distance = 0

        for move in move_options:
            new_position = self.get_new_position(prey, move)
            min_distance_to_predator = min(self.distance(new_position, predator) for predator in self.predators)

            if min_distance_to_predator > longest_distance:
                longest_distance = min_distance_to_predator
                best_move = move
        
        if best_move:
            new_position = self.get_new_position(prey, best_move)
            prey[0], prey[1] = new_position[0], new_position[1]

    def check_capture(self):
        for predator in self.predators:
            for prey in self.prey:
                if self.is_collision(predator, prey):
                    self.prey.remove(prey)
                    print(f"🔥 Prey at {prey} was captured by predator at {predator}!")

    def render(self):
        grid = np.zeros((self.grid_size, self.grid_size))

        for predator in self.predators:
            grid[predator[1], predator[0]] = 1

        for prey in self.prey:
            grid[prey[1], prey[0]] = 2
        
        self.ax.clear()
        self.ax.matshow(grid, cmap='coolwarm')

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if grid[j, i] == 1:
                    self.ax.text(i, j, 'P', va='center', ha='center', color='black', fontsize=10)
                elif grid[j, i] == 2:
                    self.ax.text(i, j, 'Y', va='center', ha='center', color='black', fontsize=10)

        plt.draw()
        plt.pause(0.1)

    def simulate(self, max_steps=100):
        print("\n=== Starting Predator-Prey Simulation ===")
        while self.prey and self.steps < max_steps:
            for predator in self.predators:
                self.move_predator(predator)
            
            for prey in self.prey:
                self.move_prey(prey)

            self.check_capture()
            self.render()

            self.steps += 1
            time.sleep(0.2)

        print(f"🏆 Simulation finished in {self.steps} steps.")
        if not self.prey:
            print("🔥 All prey have been captured!")
        else:
            print("❌ Some prey survived.")

        plt.ioff()
        plt.show()

# Environment settings
grid_size = 10
num_predators = 3
num_prey = 2

# Run the simulation
env = PredatorPreyEnvironment(grid_size, num_predators, num_prey)
env.simulate()
