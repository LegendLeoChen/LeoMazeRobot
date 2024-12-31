from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from DQN import *
import threading
import torch

mutex = QMutex()
crosspos = 15  # 坐标图左上角位置（横纵坐标一样）

x = 0
y = 0
next_state = None


class Drawing(QWidget):
    global x, y, next_state

    def __init__(self):
        super(Drawing, self).__init__()
        self.timer = QTimer()
        self.setMinimumSize(500, 500)
        self.maze = [  # 原迷宫
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0.5, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
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

    def replace_position(self, list1, list2):
        # 找到 list1 中 0.5 的位置
        pos1 = None
        for i in range(len(list1)):
            for j in range(len(list1[0])):
                if list1[i][j] == 0.5:
                    pos1 = (i, j)
                    break
            if pos1:
                break
        # 找到 list2 中 0.5 的位置
        pos2 = None
        for i in range(len(list2)):
            for j in range(len(list2[0])):
                if list2[i][j] == 0.5:
                    pos2 = (i, j)
                    break
            if pos2:
                break
        # 如果找到位置，进行交换
        if pos1 and pos2:
            list1[pos1[0]][pos1[1]] = 0  # 将第一个列表中的 0.5 替换为 0
            list1[pos2[0]][pos2[1]] = 0.5  # 将第二个列表中的 0.5 位置赋值给第一个列表
        return list1

    def update_maze(self, next_state):  # 更新绘图
        self.maze = self.replace_position(self.maze, next_state)
        self.repaint()

    def paintEvent(self, e):
        qp = QPainter(self)
        qp.begin(self)
        self.drawLines(qp, self.maze)
        qp.end()

    def drawLines(self, qp, maze):  # 设置迷宫每个格子的大小
        cell_size = 60  # 每个格子的宽和高

        # 遍历迷宫矩阵
        for j, row in enumerate(maze):
            for i, cell in enumerate(row):
                if cell == 1:  # 墙体绘制，实心线
                    if j % 2 == 0 and i % 2 == 0:
                        continue
                    elif j % 2 == 0:    # 横
                        qp.setPen(QPen(QColor(0, 0, 0), 5))
                        qp.drawLine(int(i / 2 * cell_size - 0.5 * cell_size), int(j / 2 * cell_size), int(i / 2 * cell_size + 0.5 * cell_size), int(j / 2 * cell_size))
                    elif i % 2 == 0:    # 竖
                        qp.setBrush(QBrush(QColor(0, 0, 0), Qt.SolidPattern))
                        qp.drawLine(int(i / 2 * cell_size), int(j / 2 * cell_size - 0.5 * cell_size), int(i / 2 * cell_size), int(j / 2 * cell_size + 0.5 * cell_size))
                elif cell == 0.5:  # 智能体绘制，实心圆形
                    qp.setBrush(QBrush(QColor(100, 0, 100), Qt.SolidPattern))
                    qp.drawEllipse(int((j - 1) / 2 * cell_size), int((i - 1) / 2 * cell_size), cell_size, cell_size)


class Example(QWidget):
    global x, y

    def __init__(self):
        super(Example, self).__init__()
        # self.postext = (str(int((x+1)/2)) + ", " + str(int((y+1)/2)))  # 坐标显示的文本
        self.initUI()
        # 利用线程，定时更新图，调用坐标显示的方法
        self.timer = QTimer()
        self.timer.start(200)

    def initUI(self):  # 总UI界面
        self._startPos = None
        self._endPos = None
        self._tracking = False
        self.setGeometry(320, 320, int(800), int(660))  # 窗体大小
        self.setWindowTitle('Position')  # 标题
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)  # 去边框
        self.label = QLabel(self)  # 坐标显示文本标签组件
        self.label.setGeometry(650, 50, int(130), int(100))
        self.label.setStyleSheet('''QLabel{
        background-color:rgb(0, 245, 255, 55);
        border: 1px solid black;
        color:rgb(0, 0, 255, 255);
        font-size:40px;
        font-weight:normal;
        font-family:Arial;}''')  # 文本样式
        # self.label.setText(self.postext)
        layout0 = QHBoxLayout()  # 总布局为左右结构
        self.wid3 = Drawing()  # 作图对象
        layout0.addWidget(self.wid3)
        self.setLayout(layout0)

        self.show()

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


app = QApplication(sys.argv)
next_state_list = []
pw = Example()


# 训练智能体
def train():
    global next_state, x, y
    env = MazeEnv()
    state_size = np.prod(env.observation_space.shape)
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    episodes = 100
    batch_size = 64

    for e in range(episodes):
        state = env.reset()
        # state = state.flatten()  # 将状态展平成一维数组
        for time in range(30000):
            action = agent.act(state)
            action_dict = ["↑", "↓", "←", "→"]
            next_state, reward, done, _ = env.step(action)
            next_state_list = next_state.tolist()
            pw.wid3.update_maze(next_state_list)
            # next_state = next_state.flatten()
            # print(f'{action_dict[action]}', end="")
            agent.remember(state, action, reward, next_state, done)
            pw.wid3.update_maze(next_state_list)
            if done:  # 打印episode、步数、当前epsilon
                print(f"episode: {e}/{episodes}, score: {time}, e: {agent.epsilon:.2}")
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        torch.save(agent.model.state_dict(), 'DQN.pth')
        torch.save(agent.target_model.state_dict(), 'DQN-target.pth')


thread1 = threading.Thread(name='t1', target=train)
thread1.start()  # 启动线程1
# QMetaObject.invokeMethod(pw.wid3, "update_maze", Qt.QueuedConnection, Q_ARG(list, next_state_list))

if __name__ == '__main__':
    sys.exit(app.exec_())
