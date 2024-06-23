from __future__ import annotations
from game import *
import time

class Node:
    def __init__(self, parent: Node, move: int) -> None:
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0
        self.ties = 0
        self.visits = 0
        self.terminal = False
        self.root = parent is None
        self.player = -1 if parent is None else -parent.player
    
    def select(self, board: np.ndarray) -> Node:
        if not self.root: move(board, self.move, self.player) # if root node, don't make a move
        if len(self.children)==0: return self
        return max([c for c in self.children], key = lambda x: (x.wins/x.visits + 2*np.sqrt(np.log(self.visits)/x.visits)) if x.visits else 0).select(board)

    def expand(self, board: np.ndarray) -> typing.List[Node]:
        if self.terminal:
            undo(board, self.move, self.player)
            return [self for _ in range(len(board[0]))] # if terminal node, do not expand, but still simulate
        for child_move in valid_moves(board):
            self.children.append(Node(self, child_move))
        for child in self.children:
            move(board, child.move, child.player)
            winner = check_winner(board)
            child.terminal = winner!=0 or len(valid_moves(board))==0
            undo(board, child.move, child.player)
        if any([c.terminal for c in self.children]): return [c for c in self.children if c.terminal]
        # if any child is terminal, only expand terminal children
        # (helps the search recognize/avoid immediate losses)
        return self.children
    
    def simulate(self, board: np.ndarray, n_sims: int) -> typing.Tuple[int,int,int]:
        return parallel(board, n_sims, self.player)
    
    def backpropagate(self, wins: int, ties: int, losses: int) -> None:
        self.wins += wins
        self.ties += ties
        self.visits += wins + ties + losses
        if self.parent is not None:
            self.parent.backpropagate(losses, ties, wins)

@nb.jit(nopython=True)
def rollout(board: np.ndarray, player: int) -> typing.Tuple[int,int,int]:
    board_copy = np.copy(board)
    simplayer = -player
    winner = check_winner(board_copy)
    moves = valid_moves(board_copy)
    while (winner == 0) and (len(moves) > 0):
        move(board_copy, np.random.choice(moves), simplayer)
        simplayer = -simplayer
        winner = check_winner(board_copy)
        moves = valid_moves(board_copy)
    return winner

@nb.jit(nb.types.Tuple((nb.int64, nb.int64, nb.int64))(nb.float64[:, :], nb.int64, nb.int8), nopython=True, fastmath=True, nogil=True, parallel=True)
def parallel(board: np.ndarray, n_sims: int, player: int) -> typing.Tuple[int,int,int]:
    wins,ties,losses = 0,0,0
    for _ in nb.prange(n_sims):
        winner = rollout(board, player)
        if winner == player: wins += 1
        elif winner == 0: ties += 1
        else: losses += 1
    return wins, ties, losses

def monte_carlo_eval(root: Node, board: np.ndarray, n_nodes: int = 1000, n_sims: int = 20) -> None:
    for _ in range(n_nodes):
        board_copy = np.copy(board)
        leaf = root.select(board_copy)
        for child in leaf.expand(board_copy):
            move(board_copy, child.move, child.player)
            wins,ties,losses = child.simulate(board_copy, n_sims)
            child.backpropagate(wins, ties, losses)
            undo(board_copy, child.move, child.player)

def find_root(node: Node) -> Node:
    while node.parent is not None:
        node = node.parent
    return node

if __name__ == '__main__':
    root = Node(None, None)
                      
    while True:
        curr = 1
        board = np.zeros([6,7])

        while check_winner(board)==0 and len(valid_moves(board))>0:
            start_time = time.time()
            monte_carlo_eval(root, board)
            end_time = time.time() 
            print(f'{root.visits:_}',"visits to root,",f'{end_time-start_time:.4f}',"more seconds")
            print_board(board)
            print(' '.join(['0' if c.visits==0 else f'{c.wins/c.visits:.4f}' for c in root.children]))

            place = input(["","Red","Blue"][curr]+"'s move: ")
            if place == 'reset':
                root.root = False
                root = find_root(root)
                root.root = True
                board = np.zeros([6,7])
                curr = 1
                continue
            place = int(place)
            while place not in valid_moves(board) and not place==-1:
                place = int(input("Invalid move. Enter move: "))
                
            root.root = False
            if place==-1:
                root = max([c for c in root.children], key = lambda x: x.wins/x.visits if x.visits else 0)
                if root.visits==0: print("No visits",root.move)
                else: print(root.move,"chosen with",f'{root.wins/root.visits:.4f}',"win rate",f'{root.visits:_}',"visits")
                move(board, root.move, curr)
            else:
                root = [c for c in root.children if c.move == place][0]
                move(board, place, curr)
            root.root = True
            curr = -curr

        print_board(board)
        print(["No one","Red","Blue"][int(check_winner(board))]+" wins!")
        root.root = False
        root = find_root(root)
        root.root = True