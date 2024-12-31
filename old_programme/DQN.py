# —*-coding:utf-8 一*一
"""
作者:LegendLeo
日期:2024年09月13日
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
            [1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
            [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
            [1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1],
            [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
            [1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
            [1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1],
            [1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
            [1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
            [1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ]
        self.start = (0, 0)  # 红点位置
        self.end = (4, 4)  # 绿点位置
        self.agent_position = list(self.start)
        self.action_space = gym.spaces.Discrete(4)  # 上下左右 4 个动作
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(10, 10), dtype=np.float32)
        self.prev_action = None  # 保存上一次的动作
        self.prev_position = None  # 保存上一次的位置
        self.prev_prev_position = None
        self.wall_hit_count = 0  # 撞墙计数
        self.swing_move_count = 0  # 重复动作计数
        self.recent_positions = []  # 记录最近n次位置
        self.n_positions = 10  # 控制记录最近几次位置

    def reset(self):
        self.agent_position = list(self.start)
        return self._get_observation()

    def step(self, action):
        col, row = [x * 2 + 1 for x in self.agent_position]
        reward = 0
        move_successful = False
        if action == 0 and self.maze[row - 1][col] == 0:  # 向上
            self.agent_position[1] -= 1
            move_successful = True
        elif action == 1 and self.maze[row + 1][col] == 0:  # 向下
            self.agent_position[1] += 1
            move_successful = True
        elif action == 2 and self.maze[row][col - 1] == 0:  # 向左
            self.agent_position[0] -= 1
            move_successful = True
        elif action == 3 and self.maze[row][col + 1] == 0:  # 向右
            self.agent_position[0] += 1
            move_successful = True
        # 终点检测
        done = tuple(self.agent_position) == self.end
        # print(action, ' ', self.agent_position, " ", move_successful)
        # 撞墙检测
        if not move_successful:
            self.wall_hit_count += 1  # 增加撞墙计数
            reward = -5.0 - 0.0 * self.wall_hit_count  # 连续撞墙增加惩罚
        else:
            self.wall_hit_count = 0  # 如果没有撞墙，重置撞墙计数
            if done:
                reward = 100  # 到达终点给予正奖励
            else:
                reward = -1.0  # 基础的每步惩罚，鼓励尽快找到终点
        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        obs = np.array(self.maze, dtype=np.float32)
        # obs = np.zeros((10, 10), dtype=np.float32)
        obs[self.agent_position[0] * 2 + 1][self.agent_position[1] * 2 + 1] = 0.5  # 智能体位置
        return obs

    def render(self):
        obs = self._get_observation()
        print("\n".join(" ".join(str(int(cell)) for cell in row) for row in obs))


# 定义 Q 网络
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        # 输入通道为 1，输出通道为 8，卷积核大小为 3×3
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        # 第二层卷积，输入通道为 8，输出通道为 16，卷积核大小为 3×3
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        # 第三层卷积，输入通道为 16，输出通道为 32，卷积核大小为 3×3
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.6)
        # 全连接层，将卷积层的输出展平成一维后，映射到 4 维的输出
        self.fc1 = nn.Linear(32 * 5 * 5, 4)

    def forward(self, x):
        # 卷积 -> 激活函数 -> 池化
        x = self.dropout(torch.relu(self.conv1(x)))
        x = self.pool1(x)  # 21 -> 10
        x = self.dropout(torch.relu(self.conv2(x)))
        x = self.pool2(x)  # 10 -> 5
        x = torch.relu(self.conv3(x))
        # 展平
        x = x.view(-1, 32 * 5 * 5)
        return self.fc1(x)


# DQN 智能体
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=4000)
        self.gamma = 0.90  # 折扣因子
        self.epsilon = 0.9  # 探索率
        self.epsilon_min = 0.0001
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.001
        self.model = DQN(state_size, action_size).cuda()
        self.target_model = DQN(state_size, action_size).cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.count = 0
        self.update_rate = 10

    def remember(self, state, action, reward, next_state, done):  # 记住轨迹
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):  # ε-贪婪算法选取动作
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)
        q_values = self.model(state.cuda())
        return torch.argmax(q_values[0]).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states, next_states, rewards, actions, dones = [], [], [], [], []
        for state, action, reward, next_state, done in minibatch:  # 更新参数
            states.append(torch.FloatTensor(state).unsqueeze(0).cuda())
            next_states.append(torch.FloatTensor(next_state).unsqueeze(0).cuda())
            rewards.append(torch.FloatTensor([reward]).cuda())
            actions.append(torch.FloatTensor([action]).cuda())
            dones.append(torch.FloatTensor([done]).cuda())
        # 将列表转换为张量
        states = torch.stack(states)
        next_states = torch.stack(next_states)
        rewards = torch.stack(rewards)
        actions = torch.stack(actions)
        dones = torch.stack(dones)
        # Q值及其目标值
        Q_expect = self.model(states).gather(1, actions.to(torch.int64))
        Q_targets = rewards + self.gamma * self.target_model(next_states).max(1)[0].view(-1, 1) * (1 - dones)
        self.optimizer.zero_grad()                      # 梯度清零
        loss = self.criterion(Q_expect, Q_targets)      # 损失
        loss.backward()                                 # 反向传播
        self.optimizer.step()                           # 参数更新
        if self.count % 5 == self.update_rate:
            self.target_model.load_state_dict(self.model.state_dict())      # 目标网络参数更新
        self.count += 1
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay          # epsilon衰减


# 训练智能体
def train():
    global next_state
    env = MazeEnv()
    state_size = np.prod(env.observation_space.shape)
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    episodes = 100
    batch_size = 32

    for e in range(episodes):
        state = env.reset()
        for time in range(10000):
            action = agent.act(state)
            action_dict = ["↑", "↓", "←", "→"]
            next_state, reward, done, _ = env.step(action)
            # print(f'{action_dict[action]}', end="")
            agent.remember(state, action, reward, next_state, done)     # 五元组加入经验回放池
            state = next_state
            if done:                                # 打印episode、步数、当前epsilon
                print(f"episode: {e}/{episodes}, score: {time}, e: {agent.epsilon:.2}")
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)            # 更新网络
