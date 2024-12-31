# —*-coding:utf-8 一*一
"""
作者:LegendLeo
日期:2024年09月13日
没有UI训练智能体走迷宫
"""
import gym
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim


# 自定义迷宫环境
class MazeEnv(gym.Env):
    def __init__(self):
        super(MazeEnv, self).__init__()
        self.maze = [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
            [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
            [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
            [1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
            [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
            [1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1],
            [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1],
            [1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
            [1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
            [1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ]
        self.start = (1, 1)  # 红点位置
        self.end = (20, 20)  # 绿点位置
        self.agent_position = list(self.start)
        self.action_space = gym.spaces.Discrete(4)  # 上下左右 4 个动作
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(21, 21), dtype=np.float32)

    def reset(self):
        self.agent_position = list(self.start)
        return self._get_observation()

    def step(self, action):
        row, col = self.agent_position

        if action == 0 and self.maze[row - 1][col] == 0:  # 向上
            self.agent_position[0] -= 1
        elif action == 1 and self.maze[row + 1][col] == 0:  # 向下
            self.agent_position[0] += 1
        elif action == 2 and self.maze[row][col - 1] == 0:  # 向左
            self.agent_position[1] -= 1
        elif action == 3 and self.maze[row][col + 1] == 0:  # 向右
            self.agent_position[1] += 1

        done = tuple(self.agent_position) == self.end
        reward = 1 if done else -0.1

        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        obs = np.zeros((21, 21), dtype=np.float32)
        obs[self.agent_position[0]][self.agent_position[1]] = 0.5  # 智能体位置
        print(self.agent_position[0], self.agent_position[1], end='')
        return obs

    def render(self):
        obs = self._get_observation()
        print("\n".join(" ".join(str(int(cell)) for cell in row) for row in obs))


# 定义 Q 网络
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# DQN 智能体
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):  # 记住轨迹
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):  # ε-贪婪算法选取动作
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state)
        return torch.argmax(q_values[0]).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:  # 更新参数
            state = torch.FloatTensor(state).unsqueeze(0)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.model(next_state)[0]).item()
            target_f = self.model(state)
            target_f[0][action] = target
            self.optimizer.zero_grad()  # 梯度清零
            loss = self.criterion(target_f, self.model(state))  # 损失
            loss.backward()  # 反向传播
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# 训练智能体
if __name__ == "__main__":
    env = MazeEnv()
    state_size = np.prod(env.observation_space.shape)
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    episodes = 100
    batch_size = 32

    for e in range(episodes):
        state = env.reset()
        state = state.flatten()  # 将状态展平成一维数组
        for time in range(1000):
            action = agent.act(state)
            action_dict = ["↑", "↓", "←", "→"]
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.flatten()
            print(f'{action_dict[action]}', end="")
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:  # 打印episode、步数、当前epsilon
                print(f"episode: {e}/{episodes}, score: {time}, e: {agent.epsilon:.2}")
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
