import math
import numpy as np
import cv2
from PIL import Image
import streamlit as st
from simpleai.search import SearchProblem, astar

# Thiết lập các chi phí di chuyển
MOVE_COST = {
    "up": 1.0,
    "down": 1.0,
    "left": 1.0,
    "right": 1.0,
}

# Bản đồ mê cung
MAZE_STRING = """
##############################
#         #              #   #
# ####    ########       #   #
#    #    #              #   #
#    ###     #####  ######   #
#      #   ###   #           #
#      #     #   #  #  #   ###
#     #####    #    #  #     #
#              #       #     #
##############################
"""

# Chuyển đổi bản đồ mê cung thành danh sách hai chiều
MAZE = [list(row) for row in MAZE_STRING.split("\n") if row]
ROWS, COLS = len(MAZE), len(MAZE[0])
CELL_SIZE = 21

# Khởi tạo hình ảnh mê cung
DARK_BLUE = np.full((CELL_SIZE, CELL_SIZE, 3), (28, 89, 56), dtype=np.uint8)  # Tường
WHITE_BLOCK = np.full((CELL_SIZE, CELL_SIZE, 3), (255, 255, 255), dtype=np.uint8)  # Đường đi
maze_image = np.ones((ROWS * CELL_SIZE, COLS * CELL_SIZE, 3), dtype=np.uint8) * 255

# Vẽ bản đồ lên hình ảnh
for row in range(ROWS):
    for col in range(COLS):
        if MAZE[row][col] == "#":
            maze_image[row * CELL_SIZE:(row + 1) * CELL_SIZE, col * CELL_SIZE:(col + 1) * CELL_SIZE] = DARK_BLUE
        else:
            maze_image[row * CELL_SIZE:(row + 1) * CELL_SIZE, col * CELL_SIZE:(col + 1) * CELL_SIZE] = WHITE_BLOCK

# Chuyển đổi sang định dạng PIL
def get_pil_image():
    converted_image = cv2.cvtColor(maze_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(converted_image)

# Lớp giải quyết bài toán mê cung
class MazeSolver(SearchProblem):
    def __init__(self, board):
        self.board = board
        self.start = None
        self.goal = None

        for y, row in enumerate(board):
            for x, cell in enumerate(row):
                if cell.lower() == "o":
                    self.start = (x, y)
                elif cell.lower() == "x":
                    self.goal = (x, y)
        super().__init__(initial_state=self.start)

    def actions(self, state):
        possible_actions = []
        for action, (dx, dy) in {
            "up": (0, -1),
            "down": (0, 1),
            "left": (-1, 0),
            "right": (1, 0),
        }.items():
            nx, ny = state[0] + dx, state[1] + dy
            if 0 <= ny < len(self.board) and 0 <= nx < len(self.board[0]) and self.board[ny][nx] != "#":
                possible_actions.append(action)
        return possible_actions

    def result(self, state, action):
        dx, dy = {
            "up": (0, -1),
            "down": (0, 1),
            "left": (-1, 0),
            "right": (1, 0),
        }[action]
        return state[0] + dx, state[1] + dy

    def is_goal(self, state):
        return state == self.goal

    def cost(self, state, action, state2):
        return MOVE_COST[action]

    def heuristic(self, state):
        return math.sqrt((state[0] - self.goal[0]) ** 2 + (state[1] - self.goal[1]) ** 2)

# Phần chính của ứng dụng Streamlit
st.title("Maze Solver")

# Hiển thị hình ảnh mê cung
st.image(get_pil_image(), caption='Maze', use_container_width=True)

# Biến để lưu điểm bắt đầu và kết thúc
if 'start' not in st.session_state:
    st.session_state.start = None
if 'goal' not in st.session_state:
    st.session_state.goal = None

# Chọn điểm bắt đầu và kết thúc bằng cách nhấp chuột
def click_event(x, y):
    if st.session_state.start is None:
        st.session_state.start = (x, y)
        maze_image[y * CELL_SIZE:(y + 1) * CELL_SIZE, x * CELL_SIZE:(x + 1) * CELL_SIZE] = (255, 0, 0)  # Màu đỏ cho điểm bắt đầu
    elif st.session_state.goal is None:
        st.session_state.goal = (x, y)
        maze_image[y * CELL_SIZE:(y + 1) * CELL_SIZE, x * CELL_SIZE:(x + 1) * CELL_SIZE] = (0, 255, 0)  # Màu xanh cho điểm kết thúc

# Hiển thị hình ảnh mê cung và cho phép nhấp chuột
col1, col2 = st.columns([1, 1])
with col1:
    x_click = st.number_input("Nhập tọa độ X (0 đến {})".format(COLS - 1), min_value=0, max_value=COLS - 1, value=0)
with col2:
    y_click = st.number_input("Nhập tọa độ Y (0 đến {})".format(ROWS - 1), min_value=0, max_value=ROWS - 1, value=0)

if st.button("Click to Set Points"):
    click_event(x_click, y_click)

# Giải mê cung nếu đã chọn đủ điểm
if st.session_state.start is not None and st.session_state.goal is not None:
    # Cập nhật mê cung với điểm bắt đầu và kết thúc
    start_x, start_y = st.session_state.start
    goal_x, goal_y = st.session_state.goal

    # Giải bài toán mê cung
    problem = MazeSolver(MAZE)
    result = astar(problem, graph_search=True)
    path = [step[1] for step in result.path()]

    # Vẽ đường đi lên hình ảnh mê cung
    for x, y in path[1:]:
        maze_image[y * CELL_SIZE:(y + 1) * CELL_SIZE, x * CELL_SIZE:(x + 1) * CELL_SIZE] = (200, 200, 200)  # Màu cho đường đi

    # Hiển thị hình ảnh mê cung với đường đi
    st.image(get_pil_image(), caption='Maze with Solution', use_container_width=True)

    # Đánh dấu lại các điểm bắt đầu và kết thúc
    maze_image[start_y * CELL_SIZE:(start_y + 1) * CELL_SIZE, start_x * CELL_SIZE:(start_x + 1) * CELL_SIZE] = (255, 0, 0)  # Màu đỏ cho điểm bắt đầu
    maze_image[goal_y * CELL_SIZE:(goal_y + 1) * CELL_SIZE, goal_x * CELL_SIZE:(goal_x + 1) * CELL_SIZE] = (0, 255, 0)  # Màu xanh cho điểm kết thúc

    # Hiển thị lại hình ảnh mê cung với điểm bắt đầu và kết thúc
    st.image(get_pil_image(), caption='Maze with Start and End', use_container_width=True)

# Nút reset
if st.button("Reset"):
    MAZE = [list(row) for row in MAZE_STRING.split("\n") if row]
    st.session_state.start = None
    st.session_state.goal = None