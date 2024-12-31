# —*-coding:utf-8 一*一
"""
作者:LegendLeo
日期:2024年09月13日
PyQT的UI、主函数
"""
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import sys
from PyQt5.QtWidgets import QApplication
from DQN import *
import threading
import torch
from tqdm import tqdm
import utils

next_state = None
end = (2, 5)            # 终点位置


class Drawing(QWidget):
    global next_state

    def __init__(self):
        super(Drawing, self).__init__()
        self.timer = QTimer()
        self.setMinimumSize(500, 500)
        self.maze = utils.load_map('map.txt')           # 加载地图

    def replace_position(self, list1, pos):
        # 找到迷宫中智能体的位置
        pos1 = None
        for i in range(len(list1)):
            for j in range(len(list1[0])):
                if list1[i][j] == 2:
                    pos1 = (i, j)
                    break
            if pos1:
                break
        list1[pos1[0]][pos1[1]] = 0                     # 抹除旧的智能体位置
        list1[pos[1] * 2 + 1][pos[0] * 2 + 1] = 2       # 新的智能体位置
        return list1

    def update_maze(self, next_state):                  # 更新迷宫绘图
        self.maze = self.replace_position(self.maze, next_state)
        self.repaint()

    def paintEvent(self, e):                # 绘图事件
        qp = QPainter(self)
        qp.begin(self)
        self.drawLines(qp, self.maze)
        qp.end()

    def drawLines(self, qp, maze):          # 具体绘制
        cell_size = 60                      # 每个格子的宽和高
        # 终点
        qp.setPen(Qt.NoPen)
        qp.setBrush(QBrush(QColor(0, 220, 0), Qt.SolidPattern))
        qp.drawRect(int(end[0] * cell_size), int(end[1] * cell_size), cell_size, cell_size)
        ''' 遍历迷宫矩阵 '''
        for j, row in enumerate(maze):
            for i, cell in enumerate(row):
                if cell == 1:  # 墙体绘制，实心线
                    if j % 2 == 0 and i % 2 == 0:
                        continue
                    elif j % 2 == 0:                # 横墙
                        qp.setPen(QPen(QColor(0, 0, 0), 5))
                        qp.drawLine(int(i / 2 * cell_size - 0.5 * cell_size), int(j / 2 * cell_size), int(i / 2 * cell_size + 0.5 * cell_size), int(j / 2 * cell_size))
                    elif i % 2 == 0:                # 竖墙
                        qp.setBrush(QBrush(QColor(0, 0, 0), Qt.SolidPattern))
                        qp.drawLine(int(i / 2 * cell_size), int(j / 2 * cell_size - 0.5 * cell_size), int(i / 2 * cell_size), int(j / 2 * cell_size + 0.5 * cell_size))
                elif cell == 2:                     # 智能体，实心圆形
                    qp.setBrush(QBrush(QColor(100, 0, 100), Qt.SolidPattern))
                    qp.drawEllipse(int((i - 1) / 2 * cell_size), int((j - 1) / 2 * cell_size), cell_size, cell_size)


class Example(QWidget):

    def __init__(self):
        super(Example, self).__init__()
        self.initUI()
        # 利用线程，定时更新图，调用坐标显示的方法（显示功能可能卡死，先关了）
        # self.timer = QTimer()
        # self.timer.start(3000)
        # self.timer.timeout.connect(self.show_data)
        self.r = 0
        self.rounds = 0
        self.epoch = 0
        self.result = 0
        self.success = 0
        self.success_test = 0

    def initUI(self):               # 总UI界面
        self._startPos = None
        self._endPos = None
        self._tracking = False
        self.setGeometry(320, 320, int(800), int(660))  # 窗体大小
        self.setWindowTitle('Position')  # 标题
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)  # 去边框
        self.label = QLabel(self)  # 坐标显示文本标签组件
        self.label.setGeometry(620, 50, int(150), int(500))
        self.label.setStyleSheet('''QLabel{
                                        background-color:rgb(0, 245, 255, 55);
                                        border: 1px solid black;
                                        color:rgb(0, 0, 255, 255);
                                        font-size:30px;
                                        font-weight:normal;
                                            font-family:Arial;
                                    }''')       # 文本样式
        self.label.setAlignment(Qt.AlignCenter)
        layout0 = QHBoxLayout()                 # 总布局为左右结构
        self.wid3 = Drawing()                   # 作图对象
        layout0.addWidget(self.wid3)
        self.setLayout(layout0)
        self.show()

    def show_data(self):
        self.label.setText(f"波数\n{self.r}/{self.rounds}\n\n迭代数\n{self.epoch}\n\n"
                           f"平均回报\n{self.result}\n\n成功次数\n{self.success}\n\n测试成功\n{self.success_test}")

    def set_data(self, r, rounds, epoch, result, success, success_test):
        self.r = r
        self.rounds = rounds
        self.epoch = epoch
        self.result = result
        self.success = success
        self.success_test = success_test

    def mouseMoveEvent(self, e: QMouseEvent):  # 重写移动事件，按住屏幕可移动窗口
        if self._tracking:
            self._endPos = e.pos() - self._startPos
            self.move(self.pos() + self._endPos)

    def mousePressEvent(self, e: QMouseEvent):
        if e.button() == Qt.LeftButton:
            self._startPos = QPoint(e.x(), e.y())
            self._tracking = True

    def mouseReleaseEvent(self, e: QMouseEvent):
        if e.button() == Qt.LeftButton:
            self._tracking = False
            self._startPos = None
            self._endPos = None


# 训练智能体
def train():
    global next_state
    env = MazeEnv()         # 定义迷宫环境
    state_size = np.prod(env.observation_space.shape)   # 状态空间，xy坐标，即2
    action_size = env.action_space.n                    # 动作空间，上下左右，即4
    agent = DQNAgent(state_size, action_size)           # 定义智能体
    batch_size = 64         # 批次大小
    episodes = 1000         # 总迭代数
    rounds = 20             # 迭代数平均分为若干波
    epoch = 0               # 记录当前迭代数
    return_list = []        # 记录每代回报
    success = 0             # 成功次数（训练）
    succ_test = 0           # 成功次数（测试）

    ''' 迭代次数分为 rounds 波统计 '''
    for i in range(rounds):
        with tqdm(total=int(episodes / rounds), desc=f'波数 {i + 1} / {rounds}') as pbar:
            ''' 每一波（一代代循环） '''
            for e in range(int(episodes / rounds)):
                epoch += 1
                episode_return = 0
                state = env.reset()
                ''' 采样和训练环节  每一代（一步步循环） '''
                for time in range(0, 30):
                    action = agent.act(state)                           # 动作
                    next_state, reward, done, _ = env.step(action)      # 回报、下一个状态、是否到达
                    episode_return += reward                            # 本次迭代总回报
                    pw.wid3.update_maze(next_state)                     # 更新迷宫UI
                    action_dict = ["↑", "↓", "←", "→"]
                    # print(f'{action_dict[action]}', end="")
                    agent.remember(state, action, reward, next_state, done)     # 更新经验回放池
                    if done:                                    # 到达终点，记录成功次数
                        success += 1
                        break
                    if len(agent.memory) > batch_size:          # 训练
                        agent.replay(batch_size)

                return_list.append(episode_return)              # 记录每代回报
                ''' 测试环节 '''
                state = env.reset()
                path = []                                       # 单次测试的路径
                for t in range(0, 8):                           # 最短步数内能否到达终点
                    action = agent.act(state, False)
                    state, _, done, _ = env.step(action)
                    if done:
                        succ_test += 1
                        if succ_test == 8:                      # 累计到达若干次，则结束训练退出
                            torch.save(agent.model.state_dict(), 'DQN.pth')         # 保存权重
                            torch.save(agent.target_model.state_dict(), 'DQN-target.pth')
                            print(f"训练完成！成功找到路径！权重已保存。步数：{t}，路径为{path}")
                            utils.draw_image(return_list)                           # 绘制回报曲线
                            return
                    path.append(action)
                ''' 更新进度条和UI显示（UI更新开了可能会卡死，所以注释了） '''
                recent_mean = int(np.mean(return_list[-20:]))                       # 近期回报均值
                # pw.set_data(r=i + 1, rounds=rounds, epoch=epoch,
                #             result=recent_mean, success=success, success_test=succ_test)
                pbar.set_postfix({'迭代数': '%d' % epoch, '平均回报': '%d' % recent_mean,
                                  '训练成功次数': '%d' % success, '测试成功次数': '%d' % succ_test})
                pbar.update(1)
    print(f"训练结束，但未找到路径。")
    utils.draw_image(return_list)                   # 绘制回报曲线


if __name__ == '__main__':
    app = QApplication(sys.argv)
    pw = Example()                  # UI线程
    thread1 = threading.Thread(name='t1', target=train)
    thread1.start()                 # 训练线程
    sys.exit(app.exec_())
