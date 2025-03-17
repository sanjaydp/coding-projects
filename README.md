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

🚀 Next-Level Enhancements:
✅ 1. Competitive Learning Between Predators
Introduce a reward hierarchy:
Predators get individual rewards for catching prey.
Add a bonus reward for teamwork (if more than one predator participates).
Add a penalty for interfering with another predator’s catch.
✅ 2. Introduce Predator "Roles"
Assign different behaviors to predators:
Chasers – Aggressive, high-reward for capture.
Blockers – Defend territory and reduce prey escape options.
Scouts – Locate prey and communicate positions to others.
✅ 3. Multi-Agent Cooperation and Competition
Add communication between predators to form temporary alliances.
Include probabilistic betrayal — predators may defect for higher individual rewards.
✅ 4. Adaptive Prey Strategy
Prey should adapt dynamically:
Form groups when threatened.
Introduce a leader-following mechanism for coordinated movement.
Prey can start “decoying” to mislead predators.
✅ 5. Memory for Prey and Predators
Implement a short-term memory:
Agents remember past actions and outcomes.
Use this to improve decision-making.
✅ 6. Reinforcement Learning with PPO (Proximal Policy Optimization)
Upgrade from DQN to PPO for more stable learning:
PPO handles continuous action spaces better.
PPO helps reduce overfitting and instability.
✅ 7. Add Logging to TensorBoard
Log the following metrics:
Average reward per episode
Average number of steps to capture
Success rate over episodes

✅ The code is now upgraded with elite-level MAS complexity:

🔥 Predator Roles – Predators now take on Chaser, Blocker, and Scout roles.
🔥 Competitive and Cooperative Rewards – Higher rewards for team play; penalties for selfish behavior.
🔥 Multi-Agent Communication – Predators share information to improve hunting strategy.
🔥 Adaptive Prey Strategy – Prey actively tries to mislead predators.
🔥 Short-Term Memory – Agents now adapt based on recent experiences.

### Final Version Features 
✅ PPO-based learning  
✅ Predator roles: Chaser, Blocker, Scout  
✅ Prey roles: Leader, Decoy  
✅ Real-time communication between predators  
✅ Dynamic reward system  
✅ Heatmap visualization of movement  

### How to Run
python multi-agent-dqn/multi_agent_predator_prey_dqn.py