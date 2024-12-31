# 导入相关包

import random
from Maze import Maze
from Runner import Runner
from QRobot import QRobot
from Maze import Maze
from Runner import Runner
import numpy as np

# 机器人移动方向
move_map = {
    'u': (-1, 0), # up
    'r': (0, +1), # right
    'd': (+1, 0), # down
    'l': (0, -1), # left
}


# 迷宫路径搜索树
class SearchTree(object):


    def __init__(self, loc=(), action='', parent=None):
        """
        初始化搜索树节点对象
        :param loc: 新节点的机器人所处位置
        :param action: 新节点的对应的移动方向
        :param parent: 新节点的父辈节点
        """

        self.loc = loc  # 当前节点位置
        self.to_this_action = action  # 到达当前节点的动作
        self.parent = parent  # 当前节点的父节点
        self.children = []  # 当前节点的子节点

    def add_child(self, child):
        """
        添加子节点
        :param child:待添加的子节点
        """
        self.children.append(child)

    def is_leaf(self):
        """
        判断当前节点是否是叶子节点
        """
        return len(self.children) == 0


def expand(maze, is_visit_m, node):
    """
    拓展叶子节点，即为当前的叶子节点添加执行合法动作后到达的子节点
    :param maze: 迷宫对象
    :param is_visit_m: 记录迷宫每个位置是否访问的矩阵
    :param node: 待拓展的叶子节点
    """
    can_move = maze.can_move_actions(node.loc)
    for a in can_move:
        new_loc = tuple(node.loc[i] + move_map[a][i] for i in range(2))
        if not is_visit_m[new_loc]:
            child = SearchTree(loc=new_loc, action=a, parent=node)
            node.add_child(child)


def back_propagation(node):
    """
    回溯并记录节点路径
    :param node: 待回溯节点
    :return: 回溯路径
    """
    path = []
    while node.parent is not None:
        path.insert(0, node.to_this_action)
        node = node.parent
    return path


def breadth_first_search(maze):
    """
    对迷宫进行广度优先搜索
    :param maze: 待搜索的maze对象
    """
    start = maze.sense_robot()
    root = SearchTree(loc=start)
    queue = [root]  # 节点队列，用于层次遍历
    h, w, _ = maze.maze_data.shape
    is_visit_m = np.zeros((h, w), dtype=np.int32)  # 标记迷宫的各个位置是否被访问过
    path = []  # 记录路径
    while True:
        current_node = queue[0]
        is_visit_m[current_node.loc] = 1  # 标记当前节点位置已访问

        if current_node.loc == maze.destination:  # 到达目标点
            path = back_propagation(current_node)
            break

        if current_node.is_leaf():
            expand(maze, is_visit_m, current_node)

        # 入队
        for child in current_node.children:
            queue.append(child)

        # 出队
        queue.pop(0)

    return path


maze = Maze(maze_size=10)
height, width, _ = maze.maze_data.shape

path_1 = breadth_first_search(maze)
print("搜索出的路径：", path_1)

for action in path_1:
    maze.move_robot(action)

if maze.sense_robot() == maze.destination:
    print("恭喜你，到达了目标点")

print(maze)


def my_search(maze):
    """
    任选深度优先搜索算法、最佳优先搜索（A*)算法实现其中一种
    :param maze: 迷宫对象
    :return :到达目标点的路径 如：["u","u","r",...]
    """

    path = []

    # -----------------请实现你的算法代码--------------------------------------
    start = maze.sense_robot()
    root = SearchTree(loc=start)
    stack = [root]  # 节点栈，用于深度优先遍历
    h, w, _ = maze.maze_data.shape
    is_visit_m = np.zeros((h, w), dtype=np.int32)  # 标记迷宫的各个位置是否被访问过
    path = []  # 记录路径

    while stack:
        current_node = stack.pop()
        is_visit_m[current_node.loc] = 1  # 标记当前节点位置已访问

        # 如果当前节点是目标点，回溯路径
        if current_node.loc == maze.destination:
            path = back_propagation(current_node)
            break

        # 如果当前节点是叶子节点，则拓展其子节点
        if current_node.is_leaf():
            expand(maze, is_visit_m, current_node)

        # 将所有未被访问过的子节点压入栈中
        for child in reversed(current_node.children):  # 反向压栈，以便保持深度优先的顺序
            if not is_visit_m[child.loc]:
                stack.append(child)

    # -----------------------------------------------------------------------
    return path


maze = Maze(maze_size=10) # 从文件生成迷宫

path_2 = my_search(maze)
print("搜索出的路径：", path_2)

for action in path_2:
    maze.move_robot(action)


if maze.sense_robot() == maze.destination:
    print("恭喜你，到达了目标点")
print(maze)

from QRobot import QRobot
from collections import deque
from torch import nn
import torch.optim as optim
import torch
import collections
from MinDQNRobot import MinDQNRobot as TorchRobot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ReplayBuffer:
    ''' 经验回放池 '''
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, action, reward, next_state, done):  # 将数据加入buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)

# 定义 Q 网络
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc5 = nn.Linear(1024, action_size)
        self.dropout = nn.Dropout(0.6)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(torch.relu(self.fc2(x)))
        return self.fc5(x)
    
class Robot(TorchRobot):
    def __init__(self, maze):
        """
        初始化 Robot 类
        :param maze:迷宫对象
        """
        super(Robot, self).__init__(maze)
        maze.set_reward(reward={
            "hit_wall": 5.0,
            "destination": -maze.maze_size ** 2.0,
            "default": 1.0,
        })
        self.maze = maze
        self.epsilon = 0
        """开启金手指，获取全图视野"""
        self.memory.build_full_view(maze=maze)
        self.loss_list = self.train()
        
    def train(self):
        loss_list = []
        batch_size = len(self.memory)

        while True:
            loss = self._learn(batch=batch_size)
            loss_list.append(loss)
            success = False
            self.reset()
            for _ in range(self.maze.maze_size ** 2 - 1):
                a, r = self.test_update()
                if r == self.maze.reward["destination"]:
                    return loss_list
    def train_update(self):
        """
        以训练状态选择动作并更新Deep Q network的相关参数
        :return :action, reward 如："u", -1
        """
#         action, reward = "u", -1.0

        # -----------------请实现你的算法代码--------------------------------------
        def state_train():
            state=self.sense_state()
            return state
        def action_train(state):
            action=self._choose_action(state)
            return action
        def reward_train(action):
            reward=self.maze.move_robot(action)
            return reward
        state = state_train()
        action = action_train(state)
        reward = reward_train(action)

        # -----------------------------------------------------------------------
        return action, reward

    def test_update(self):
        """
        以测试状态选择动作并更新Deep Q network的相关参数
        :return : action, reward 如："u", -1
        """
#         action, reward = "u", -1.0

        # -----------------请实现你的算法代码--------------------------------------
        def state_test():
            state = torch.from_numpy(np.array(self.sense_state(), dtype=np.int16)).float().to(self.device)
            return state
        state = state_test()
        self.eval_model.eval()
        with torch.no_grad():
            q_value = self.eval_model(state).cpu().data.numpy()
        def action_test(q_value):
            action=self.valid_action[np.argmin(q_value).item()]
            return action
        def reward_test(action):
            reward=self.maze.move_robot(action)
            return reward
        action = action_test(q_value)
        reward = reward_test(action)
        # -----------------------------------------------------------------------
        return action, reward


"""  Deep Qlearning 算法相关参数： """

epoch = 30  # 训练轮数
maze_size = 3  # 迷宫size
training_per_epoch = 20

""" 使用 DQN 算法训练 """

g = Maze(maze_size=maze_size)
r = Robot(g)
runner = Runner(r)
runner.run_training(epoch, training_per_epoch)

# 生成训练过程的gif图, 建议下载到本地查看；也可以注释该行代码，加快运行速度。
# runner.generate_gif(filename="results/dqn_size10.gif")
runner.plot_results()