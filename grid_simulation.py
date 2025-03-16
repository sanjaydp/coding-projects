import random

class GridEnvironment:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.agent1 = [0, 0]  # Agent 1 start position
        self.agent2 = [grid_size - 1, grid_size - 1]  # Agent 2 start position

    def get_new_position(self, agent, move):
        """Calculate the new position based on the move"""
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
    
    def is_valid_move(self, agent, new_position, other_agent):
        """Check if the move is valid (no collision)"""
        if new_position == other_agent:
            return False
        return True
    
    def move_agent(self, agent, other_agent):
        """Attempt to move agent without causing a collision"""
        move_options = ["UP", "DOWN", "LEFT", "RIGHT"]
        random.shuffle(move_options)  # Shuffle to introduce some randomness
        
        for move in move_options:
            new_position = self.get_new_position(agent, move)
            if self.is_valid_move(agent, new_position, other_agent):
                agent[0], agent[1] = new_position[0], new_position[1]
                return move
        
        return "STAY"  # If no valid move, stay in place

    def simulate(self, steps=10):
        for _ in range(steps):
            move1 = self.move_agent(self.agent1, self.agent2)
            move2 = self.move_agent(self.agent2, self.agent1)
            print(f"Agent 1: {self.agent1}, Moved {move1}")
            print(f"Agent 2: {self.agent2}, Moved {move2}")
            print("-----------")

# Run the multi-agent simulation
env = GridEnvironment(5)
env.simulate(10)
