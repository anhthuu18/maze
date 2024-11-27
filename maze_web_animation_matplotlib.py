import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


from PIL import Image, ImageDraw
import math
from simpleai.search import SearchProblem, astar

import streamlit as st
import streamlit.components.v1 as components
from streamlit_drawable_canvas import st_canvas

# Define the map
MAP = """
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

# Convert map to a list
MAP = [list(x) for x in MAP.split("\n") if x]

# Define cost of moving around the map
cost_regular = 1.0
cost_diagonal = 1.7

# Create the cost dictionary
COSTS = {
    "up": cost_regular,
    "down": cost_regular,
    "left": cost_regular,
    "right": cost_regular
}


# Class containing the methods to solve the maze
class MazeSolver(SearchProblem):
    # Initialize the class 
    def __init__(self, board):
        self.board = board
        self.goal = (0, 0)

        for y in range(len(self.board)):
            for x in range(len(self.board[y])):
                if self.board[y][x].lower() == "o":
                    self.initial = (x, y)
                elif self.board[y][x].lower() == "x":
                    self.goal = (x, y)

        super(MazeSolver, self).__init__(initial_state=self.initial)

    # Define the method that takes actions
    # to arrive at the solution
    def actions(self, state):
        actions = []
        for action in COSTS.keys():
            newx, newy = self.result(state, action)
            if self.board[newy][newx] != "#":
                actions.append(action)

        return actions

    # Update the state based on the action
    def result(self, state, action):
        x, y = state

        if action.count("up"):
            y -= 1
        if action.count("down"):
            y += 1
        if action.count("left"):
            x -= 1
        if action.count("right"):
            x += 1

        new_state = (x, y)

        return new_state

    # Check if we have reached the goal
    def is_goal(self, state):
        return state == self.goal

    # Compute the cost of taking an action
    def cost(self, state, action, state2):
        return COSTS[action]

    # Heuristic that we use to arrive at the solution
    def heuristic(self, state):
        x, y = state
        gx, gy = self.goal

        return math.sqrt((x - gx) ** 2 + (y - gy) ** 2)

W = 21
st.title('Tìm đường trong mê cung')

if "dem" not in st.session_state:
    st.session_state["dem"] = 0

if "points" not in st.session_state:
    st.session_state["points"] = []

if "bg_image" not in st.session_state:
    bg_image = Image.open("maze.bmp")
    st.session_state["bg_image"] = bg_image

canvas_result = st_canvas(
        stroke_width = 1,
        stroke_color = '',
        background_image = st.session_state["bg_image"],
        height = 210,
        width = 630,
        drawing_mode = "point",
        point_display_radius = 0,
        display_toolbar = False,
)

if  st.session_state["dem"] == 2:
    _, col2, _, col4, _, _ = st.columns(6)

    if col2.button('Directtion'):
        if "directed" not in st.session_state:
            st.session_state["directed"] = True
            x1 = st.session_state["points"][0][0]
            y1 = st.session_state["points"][0][1]

            x2 = st.session_state["points"][1][0]
            y2 = st.session_state["points"][1][1]

            MAP[y1][x1] = 'o'
            MAP[y2][x2] = 'x'
            problem = MazeSolver(MAP)
            # Run the solver
            result = astar(problem, graph_search=True)

            # Extract the path
            path = [x[1] for x in result.path()]

            print(path)
            st.session_state["path"] = path
            frame = st.session_state["bg_image"].copy()
            for p in path:
                x = p[0]
                y = p[1]
                toa_do_x = x*W+2
                toa_do_y = y*W+2
                frame_temp = ImageDraw.Draw(frame)
                frame_temp.ellipse([toa_do_x, toa_do_y, toa_do_x + W-4, toa_do_y + W-4],
                fill = "#FF00FF", outline ="#FF00FF")                    
                frame = frame.copy()
            st.session_state["bg_image"] = frame
            st.rerun()           
    if col4.button('Animation'):
        if "directed" in st.session_state:
            path = st.session_state["path"]
            n = len(path)
            px = []
            py = []
            for p in path:
                px.append(p[0]*W + W // 2 - 2)
                py.append(p[1]*W + W // 2 - 2)

            im = plt.imread("maze.bmp")
            fig, ax = plt.subplots()

            image = ax.imshow(im)
            dest, = ax.plot(px[-1:], py[-1:],"ro", markersize = 10)

            red_square, = ax.plot([],[],"ro", markersize = 10)

            def init():
                return image, dest, red_square 

            def animate(i):
                red_square.set_data(px[:i+1], py[:i+1])
                return image, dest, red_square 

            anim = FuncAnimation(fig, animate, frames = n, interval = 500, 
                                init_func=init, repeat = False, blit = True)

            components.html(anim.to_jshtml(), height = 550)
    _, col2, _ = st.columns(3)
    col2.text('Nhấn Ctrl-R để chạy lại')


if canvas_result.json_data is not None:
    lst_points = canvas_result.json_data["objects"] 
    if len(lst_points) > 0:
        # print(lst_points)
        px = lst_points[-1]['left']
        py = lst_points[-1]['top']

        x = int(px) // W
        y = int(py) // W

        print('(%d, %d)' % (x, y))
        if MAP[y][x] != '#':
            if st.session_state["dem"] < 2:
                st.session_state["dem"] = st.session_state["dem"] + 1
                print('True')
                print('dem', st.session_state["dem"])
                if st.session_state["dem"] == 1:
                    toa_do_x = x*W+2
                    toa_do_y = y*W+2
                    frame = st.session_state["bg_image"].copy()
                    frame_temp = ImageDraw.Draw(frame)
                    frame_temp.ellipse([toa_do_x, toa_do_y, toa_do_x + W-4, toa_do_y + W-4],
                    fill = "#FF00FF", outline ="#FF00FF")                    
                    st.session_state["bg_image"] = frame
                    st.session_state["points"].append((x,y))
                    st.rerun()
                elif st.session_state["dem"] == 2:
                    toa_do_x = x*W+2
                    toa_do_y = y*W+2
                    frame = st.session_state["bg_image"].copy()
                    frame_temp = ImageDraw.Draw(frame)
                    frame_temp.ellipse([toa_do_x, toa_do_y, toa_do_x + W-4, toa_do_y + W-4],
                    fill = "#FF0000", outline ="#FF0000")                    
                    st.session_state["bg_image"] = frame
                    st.session_state["points"].append((x,y))
                    st.rerun()
        else:
            print('False')

