## 1. Predator-Prey Simulation
- Simulation of predators trying to catch prey using cooperative strategy.


This code: 
✅ Implements cooperative predators
✅ Uses Manhattan distance for pathfinding
✅ Uses communication between predators
✅ Real-time updates with matplotlib

🔎 Features Added:
✅ Cooperative behavior for predators
✅ Prey attempts to evade capture strategically
✅ Real-time updates and visualization
✅ Dynamic pathfinding using Manhattan distance
✅ Event-based termination (capture or survival)

### How to Run
python multi-agent-predator-prey/predator_prey.py


## 2. Predator-Prey with Q-Learning
- Multi-agent reinforcement learning using Q-Learning for predator-prey simulation.

🔥 Why RL?
✅ Reinforcement Learning allows agents to learn from experience instead of relying on pre-coded strategies.
✅ It introduces dynamic, adaptive behavior as agents improve based on feedback.
✅ Multi-Agent RL (MARL) is a hot field in robotics, game AI, and autonomous systems.

🏆 New Features to Add:
✅ Q-Learning for decision-making (agents learn from rewards and penalties).
✅ Shared state for cooperative learning among predators.
✅ Exploration vs. Exploitation – Encourage agents to explore the environment early and exploit learned knowledge later.
✅ Custom Reward System – Reward successful capture, penalize collisions.
✅ Training Phase – Run a training loop for agents to learn better strategies over time.

### How to Run
python multi-agent-rl/predator_prey_qlearning.py

## 3. Advanced Multi-Agent Predator-Prey Simulation with DQN
- Multi-agent system using Deep Q-Learning (DQN) with communication, flocking, and adaptive prey behavior.

✅ Why DQN?
Handle larger state-action spaces using neural networks.
Handle continuous state spaces better than tabular methods.

✅ Why Flocking?
Flocking creates realistic behavior based on three rules:
Alignment – Move in the same direction as nearby agents.
Cohesion – Move toward the average position of nearby agents.
Separation – Avoid getting too close to other agents.

✅ Why Adaptive Prey?
Make prey smarter by using pathfinding and grouping.
Prey can also learn using SARSA or DQN.

✅ Why Team Rewards?
Encourage cooperation over selfish behavior.
Higher reward when multiple predators contribute to catching prey.

🚀 Next-Level Enhancements:
✅ 1. SARSA-Based Prey Learning
Prey will no longer move randomly — it will learn using SARSA (State-Action-Reward-State-Action).
This introduces competition between learning predators and learning prey.
✅ 2. Reward Balancing for Cooperation vs Individual Reward
Predators will receive higher rewards for teamwork.
Adjust reward scaling to favor cooperative behavior.
✅ 3. Heatmap Visualization
Show predator and prey movements over time using a heatmap.
Track areas with higher activity and hunting success rates.
✅ 4. TensorBoard Logging
Track average reward, episode length, and prey caught over time.
Monitor training stability and learning rate.

### How to Run
python multi-agent-dqn/multi_agent_predator_prey_dqn.py