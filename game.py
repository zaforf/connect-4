import numpy as np
import numba as nb
from numba import jit
import typing
from termcolor import colored

@jit(nopython=True)
def check_winner(board: np.ndarray) -> int:
    rows, cols = board.shape
    # horizontals
    for r in range(rows):
        for c in range(cols-3):
            if abs(np.sum(board[r][c:c+4]))==4:
                return board[r][c]
    # verticals
    for r in range(rows-3):
        for c in range(cols):
            if abs(np.sum(board[r:r+4,c]))==4:
                return board[r][c]
    for r in range(rows-3):
        for c in range(cols-3):
            # \
            if abs(np.trace(board[r:r+4,c:c+4]))==4:
                return board[r][c]
            # /
            if abs(np.trace(board[r:r+4,c:c+4][::-1]))==4:
                return board[r+3][c]
    return 0

@jit(nopython=True)
def valid_moves(board: np.ndarray) -> np.ndarray:
    return np.where(board[0] == 0)[0]

@jit(nopython=True)
def move(board: np.ndarray, col: int, player: int) -> None:
    board[np.max(np.where(board[:,col] == 0)[0])][col]=player

@jit(nopython=True)
def undo(board: np.ndarray, col: int, player: int) -> None:
    board[np.min(np.where(board[:,col] != 0)[0])][col]=0

def print_board(board):
    colors = ['white','red','blue']
    for row in board.astype(np.int8):
        print(''.join(colored('●',colors[i]) for i in row))
