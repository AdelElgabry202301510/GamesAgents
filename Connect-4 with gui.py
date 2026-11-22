import tkinter as tk
from tkinter import messagebox
import numpy as np
import copy

# Constants

ROWS = 6
COLS = 7
EMPTY = ' '

CELL_SIZE = 80
RADIUS = 30

COLOR_MAP = {'x': 'red', 'o': 'blue', EMPTY: 'white'}
MODELS = ["Minimax", "Alpha-Beta"]

# Game Logic

def empty_board():
    return [[EMPTY for _ in range(COLS)] for _ in range(ROWS)]

def actions(board):
    return [c for c in range(COLS) if board[0][c] == EMPTY]

def result(state, player, action):
    state = copy.deepcopy(state)
    for r in reversed(range(ROWS)):
        if state[r][action] == EMPTY:
            state[r][action] = player
            break
    return state

def other(player):
    return 'o' if player == 'x' else 'x'

def check_board(board):
    for r in range(ROWS):
        for c in range(COLS - 3):
            w = [board[r][c+i] for i in range(4)]
            if w[0] != EMPTY and len(set(w)) == 1:
                return w[0]

    for r in range(ROWS - 3):
        for c in range(COLS):
            w = [board[r+i][c] for i in range(4)]
            if w[0] != EMPTY and len(set(w)) == 1:
                return w[0]

    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            w = [board[r+i][c+i] for i in range(4)]
            if w[0] != EMPTY and len(set(w)) == 1:
                return w[0]

    for r in range(3, ROWS):
        for c in range(COLS - 3):
            w = [board[r-i][c+i] for i in range(4)]
            if w[0] != EMPTY and len(set(w)) == 1:
                return w[0]

    if all(board[0][c] != EMPTY for c in range(COLS)):
        return 'd'
    return 'n'

def terminal(state):
    return check_board(state) != 'n'

def utility(state, player='x'):
    goal = check_board(state)
    if goal == player:
        return 1
    if goal == other(player):
        return -1
    return 0

# AI Models

def minimax_player(board, player, depth=3):

    def maxv(s, d):
        if terminal(s) or d == 0:
            return utility(s, player), None
        v, move = -np.inf, None
        for a in actions(s):
            val, _ = minv(result(s, player, a), d-1)
            if val > v:
                v, move = val, a
        return v, move

    def minv(s, d):
        if terminal(s) or d == 0:
            return utility(s, player), None
        v, move = np.inf, None
        for a in actions(s):
            val, _ = maxv(result(s, other(player), a), d-1)
            if val < v:
                v, move = val, a
        return v, move

    return maxv(board, depth)[1] if player == 'x' else minv(board, depth)[1]

def alphabeta_player(board, player, depth=4):

    def maxv(s, a, b, d):
        if terminal(s) or d == 0:
            return utility(s, player), None
        v, move = -np.inf, None
        for m in actions(s):
            val, _ = minv(result(s, player, m), a, b, d-1)
            if val > v:
                v, move = val, m
            a = max(a, v)
            if a >= b:
                break
        return v, move

    def minv(s, a, b, d):
        if terminal(s) or d == 0:
            return utility(s, player), None
        v, move = np.inf, None
        for m in actions(s):
            val, _ = maxv(result(s, other(player), m), a, b, d-1)
            if val < v:
                v, move = val, m
            b = min(b, v)
            if a >= b:
                break
        return v, move

    return maxv(board, -np.inf, np.inf, depth)[1] if player == 'x' else minv(board, -np.inf, np.inf, depth)[1]

def get_model(name):
    return minimax_player if name == "Minimax" else alphabeta_player




# GUI

class ConnectFourGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Connect Four AI")

        self.board = empty_board()
        self.turn = 'x'
        self.game_over = False

        # Controls
        ctrl = tk.Frame(root)
        ctrl.pack()

        self.mode = tk.StringVar(value="Human vs AI")
        tk.OptionMenu(ctrl, self.mode, "Human vs AI", "AI vs AI").grid(row=0, column=0)

        self.p1_model = tk.StringVar(value="Alpha-Beta")
        self.p2_model = tk.StringVar(value="Minimax")

        tk.OptionMenu(ctrl, self.p1_model, *MODELS).grid(row=0, column=1)
        tk.OptionMenu(ctrl, self.p2_model, *MODELS).grid(row=0, column=2)

        tk.Button(ctrl, text="Start Game", command=self.start_game).grid(row=0, column=3)

        # Board
        self.canvas = tk.Canvas(root, width=COLS*CELL_SIZE, height=ROWS*CELL_SIZE, bg='yellow')
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.human_move)

        self.draw_board()

    # Game Flow

    def start_game(self):
        self.board = empty_board()
        self.turn = 'x'
        self.game_over = False
        self.draw_board()

        if self.mode.get() == "AI vs AI":
            self.root.after(500, self.ai_vs_ai)

    def draw_board(self):
        self.canvas.delete("all")
        for r in range(ROWS):
            for c in range(COLS):
                x, y = c*CELL_SIZE, r*CELL_SIZE
                self.canvas.create_rectangle(x, y, x+CELL_SIZE, y+CELL_SIZE, fill='yellow')
                self.canvas.create_oval(
                    x+CELL_SIZE//2-RADIUS, y+CELL_SIZE//2-RADIUS,
                    x+CELL_SIZE//2+RADIUS, y+CELL_SIZE//2+RADIUS,
                    fill=COLOR_MAP[self.board[r][c]]
                )

    # Human vs AI

    def human_move(self, event):
        if self.game_over or self.mode.get() != "Human vs AI" or self.turn != 'x':
            return

        col = event.x // CELL_SIZE
        if col not in actions(self.board):
            return

        self.board = result(self.board, 'x', col)
        self.turn = 'o'
        self.draw_board()

        if self.end_check():
            return

        self.root.after(400, self.ai_move)

    def ai_move(self):
        if self.game_over:
            return

        model = get_model(self.p2_model.get())
        col = model(self.board, 'o')

        self.board = result(self.board, 'o', col)
        self.turn = 'x'
        self.draw_board()

        self.end_check()

    # AI vs AI

    def ai_vs_ai(self):
        if self.game_over:
            return

        model_name = self.p1_model.get() if self.turn == 'x' else self.p2_model.get()
        model = get_model(model_name)

        col = model(self.board, self.turn)
        self.board = result(self.board, self.turn, col)
        self.draw_board()

        if self.end_check():
            return

        self.turn = other(self.turn)
        self.root.after(500, self.ai_vs_ai)

    # End Game

    def end_check(self):
        outcome = check_board(self.board)
        if outcome != 'n':
            self.game_over = True

            if outcome == 'x':
                msg = "Player 1 Wins 🔴"
            elif outcome == 'o':
                msg = "Player 2 Wins 🔵"
            else:
                msg = "Draw!"

            messagebox.showinfo("Game Over", msg)
            return True
        return False

# Run

if __name__ == "__main__":
    root = tk.Tk()
    ConnectFourGUI(root)
    root.mainloop()
