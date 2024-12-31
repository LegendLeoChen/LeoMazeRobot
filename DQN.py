# —*-coding:utf-8 一*一
"""
作者:LegendLeo
日期:2024年09月13日
迷宫环境、DQN智能体、DQN网络
"""
import gym
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import utils


# 定义迷宫环境
class MazeEnv(gym.Env):
    def __init__(self):
        super(MazeEnv, self).__init__()
        self.maze = utils.load_map('map.txt')           # 加载迷宫地图
        self.start = (0, 0)                             # 起点位置
        self.end = (2, 5)                               # 终点位置
        self.agent_position = list(self.start)          # 当前智能体位置
        self.action_space = gym.spaces.Discrete(4)      # 动作空间 上下左右 4个动作
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1, 2), dtype=np.float32)  # 状态空间 xy坐标 2
        self.explored = torch.zeros(10, 10)             # 访问计数（到过的位置就+1）

    def reset(self):                                    # 重置智能体位置、访问计数清零
        self.agent_position = list(self.start)
        self.explored = torch.zeros(10, 10)
        return self._get_observation()

    def step(self, action):                             # 动作并获取回报
        col, row = [x * 2 + 1 for x in self.agent_position]
        reward = 0
        ''' 移动操作 '''
        if action == 0 and self.maze[row - 1][col] == 0:    # 向上
            self.agent_position[1] -= 1
        elif action == 1 and self.maze[row + 1][col] == 0:  # 向下
            self.agent_position[1] += 1
        elif action == 2 and self.maze[row][col - 1] == 0:  # 向左
            self.agent_position[0] -= 1
        elif action == 3 and self.maze[row][col + 1] == 0:  # 向右
            self.agent_position[0] += 1
        ''' 探索奖励 '''
        if not self.explored[self.agent_position[0], self.agent_position[1]]:   # 到新位置给小奖励
            reward += 20.0
        else:                       # 重复到达给惩罚
            reward -= 10.0 + 5.0 * self.explored[self.agent_position[0], self.agent_position[1]]
        self.explored[self.agent_position[0], self.agent_position[1]] += 1      # 寻访计数
        ''' 终点检测 '''
        done = tuple(self.agent_position) == self.end
        ''' 常规奖励 '''
        if done:
            reward += 20000         # 到达终点给予大量正奖励
        else:
            reward -= 5.0           # 基础的每步惩罚，鼓励尽快找到终点
        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        return self.agent_position

    def render(self):
        obs = self._get_observation()
        print("\n".join(" ".join(str(int(cell)) for cell in row) for row in obs))


# 定义 Q 网络
class DQN(nn.Module):
    def __init__(self, state_size: int, action_size: int, seed: int):
        super(DQN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input_hidden = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(False),
            nn.Linear(128, 256),
            nn.ReLU(False),
            nn.Linear(256, 128),
            nn.ReLU(False),
            nn.Linear(128, 128),
            nn.ReLU(False),
        )
        self.final_fc = nn.Linear(128, action_size)

    def forward(self, state):
        x = self.input_hidden(state)
        return self.final_fc(x)


# 定义 DQN 智能体
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1000)    # 经验回放池
        self.gamma = 0.90                   # 折扣因子
        self.epsilon = 1                    # 探索率
        self.epsilon_min = 0.1              # 最小探索率
        self.epsilon_decay = 0.9995         # 探索率衰减
        self.learning_rate = 0.002          # 网络的学习率
        self.model = DQN(state_size, action_size, 42).cuda()            # Q网络
        self.target_model = DQN(state_size, action_size, 42).cuda()     # 目标网络
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.count = 0                      # 计数，用于更新目标网络
        self.update_rate = 10               # 更新目标网络频率

    def remember(self, state, action, reward, next_state, done):    # 新的数据加入经验回放池
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, epsilon=True):                         # ε-贪婪算法选取动作
        if np.random.rand() <= self.epsilon and epsilon:        # 随机
            return np.random.randint(self.action_size)
        self.model.eval()
        state = torch.tensor([state], dtype=torch.float)
        with torch.no_grad():
            q_values = self.model(state.cuda())                 # 贪婪
        return q_values.argmax().item()

    def replay(self, batch_size):                   # 训练学习
        if len(self.memory) < batch_size:
            return
        ''' 采样 + 数据处理 '''
        minibatch = random.sample(self.memory, batch_size)
        states, next_states, rewards, actions, dones = [], [], [], [], []
        for state, action, reward, next_state, done in minibatch:
            states.append(torch.FloatTensor(state).cuda())
            next_states.append(torch.FloatTensor(next_state).cuda())
            rewards.append(torch.FloatTensor([reward]).cuda())
            actions.append(torch.FloatTensor([action]).cuda())
            dones.append(torch.FloatTensor([done]).cuda())
        states = torch.stack(states)
        next_states = torch.stack(next_states)
        rewards = torch.stack(rewards)
        actions = torch.stack(actions)
        dones = torch.stack(dones)
        ''' 前向传播 + Q值及其目标值计算 '''
        self.model.train()
        self.target_model.eval()
        Q_expect = self.model(states).gather(1, actions.to(torch.int64))
        Q_targets = rewards + self.gamma * self.target_model(next_states).max(1)[0].view(-1, 1) * (1 - dones)
        ''' 反向传播 + 参数更新 '''
        self.optimizer.zero_grad()                      # 梯度清零
        loss = self.criterion(Q_expect, Q_targets)      # 损失
        loss.backward()                                 # 反向传播
        self.optimizer.step()                           # 参数更新
        if self.count % self.update_rate == 0:          # 目标网络参数更新
            self.target_model.load_state_dict(self.model.state_dict())
        self.count += 1
        if self.epsilon > self.epsilon_min:             # epsilon衰减
            self.epsilon *= self.epsilon_decay
