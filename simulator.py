
import numpy as np
import torch
import enum
from enum import Enum

def fetch_legal_actions():
    pass

def piece_distance(x1, y1, x2, y2):
    return int(np.sqrt((x2 - x1) ** 2 + (y2 - y1)**2))

class CheckerBoard:
    def __init__(self):
        self.board_dim = 10
        self.board = [[(i+j + 1) % 2 for i in range(self.board_dim)] for j in range(self.board_dim)]
        self.whites = np.array([[1 if j <= 2 and self.board[i][j] == 0 else 0 for i in range(self.board_dim)] for j in range(self.board_dim)])
        self.blacks = np.array([[1 if j >= (self.board_dim - 3) and self.board[i][j] == 0 else 0 for i in range(self.board_dim)] for j in range(self.board_dim)])
        self.rewards = {
            "simple_move" : 0,
            "capture_piece" : 0,
            "invalid_move" : -5,
            "lose_game" : -1,
            "win_game" : 1
        }

    def as_tensor(self, device="cpu"):
        whites_tensor = torch.from_numpy(self.whites, device=device).unsqueeze(0)
        blacks_tensor = torch.from_numpy(self.blacks, device=device).unsqueeze(0)
        total_state = torch.cat([whites_tensor, blacks_tensor], dim=0)
        return total_state


    def check_bounds(self,x, y):
        if x not in range(self.board_dim) or y not in range(self.board_dim):
            return False
        return True

    # def check_valid_move(self, src_x, src_y, dest_x, dest_y, player):
    #     # given a source board, ie which piece to lift,
    #     # and a dest_board, ie where to place piece
    #     # determine if that move is legal

    #     player_board = self.whites if player == "white" else self.blacks
    #     opponent_board = self.whites if player == "black" else self.blacks
        
    #     # we don't have a piece at that pos so we can't move it
    #     if player_board[src_x][src_y] == 0.0:
    #         return False
        
    #     # if our destination is occupied, not allowed to move there
    #     if player_board[dest_x][dest_y] == 1.0 or opponent_board[dest_x][dest_y] == 1.0:
    #         return False

    #     dist = piece_distance(src_x, src_y, dest_x, dest_y)
    #     if dist not in [1, 2]:
    #         return False
    #     if dist == 2:
    #         # get midpoint
    #         mid_x = (src_x + dest_x) // 2
    #         mid_y = (src_y + dest_y) // 2
    #         # you cannot move two steps unless you are capturing
    #         if opponent_board[mid_x][mid_y] == 0.0:
    #             return False
    #     if dist == 1:
    #         # can't move down, ie forwards for white
    #         if player == "white" and dest_y < src_y:
    #             return False
    #         # can't move up ie backwards for black
    #         if player == "black" and dest_y > src_y:
    #             return False

    #     return True

    def check_valid_move(self, player_board, opponent_board, x, y):
        return self.check_bounds(x, y) and opponent_board[x][y] == 0 and player_board[x][y] == 0


    # i really need to optimise this 
    def get_valid_actions(self, player):
        player_board = self.whites if player == "white" else self.blacks
        opponent_board = self.whites if player == "black" else self.blacks
        
        valid_actions = []
        for i in range(self.board_dim):
            for j in range(self.board_dim):
                if player_board[i][j] == 0:
                    continue
                
                simple_move_positions = [(i+1, j+1), (i-1, j+1)]
                simple_move_positions = filter(lambda pos: self.check_valid_move, simple_move_positions)
                capture_move_positions = [(i+2, j+2), (i-2, j+2), (i+2, j-2), (i-2, j-2)]
                capture_move_positions = filter(lambda pos: self.check_valid_move, capture_move_positions)
                
                for simple_position in simple_move_positions:
                    x,y = simple_position
                    src_board = np.zeros_like(player_board)
                    dest_board = np.zeros_like(player_board)
                    src_board[i][j] = 1.0
                    dest_board[x][y] = 1.0
                    action_cat = np.concatenate([src_board, dest_board])
                    valid_actions.append(action_cat)
                
                for capture_position in capture_move_positions:
                    x,y = capture_position
                    mid_x = (i+x) // 2
                    mid_y = (j+y) // 2
                    if opponent_board[mid_x][mid_y] == 1.0:
                        src_board = np.zeros_like(player_board)
                        dest_board = np.zeros_like(player_board)
                        src_board[i][j] = 1.0
                        dest_board[x][y] = 1.0
                        action_cat = np.concatenate([src_board, dest_board])
                        valid_actions.append(action_cat)

        return valid_actions

    def execute(self, src_board, dest_board, player):
        src_x,src_y = np.unravel_index(src_board.argmax(), src_board.shape) # (i, j)
        dest_x, dest_y = np.unravel_index(dest_board.argmax(), dest_board.shape) # (i, j) 

        if not self.check_valid_move(src_x,):
            raise Exception("Error")
 
        pass

    
    def game_over(self, board, opposing_board):
        if np.all(board == 0): 
            return self.rewards["lose_game"]
        if np.all(opposing_board == 0):
            return self.rewards["win_game"]
        return self.rewards["simple_move"]

    def __repr__(self):
        final_str = ""
        final_str += "Board State\n"
        for arr in self.board:
            final_str += "".join(str(x) for x in arr) + "\n"

        final_str += "\n"
        final_str += "White\n"
        for arr in self.whites:
            final_str += "".join(str(x) for x in arr) + "\n"
        
        final_str += "\n"

        final_str += "Black\n"
        for arr in self.blacks:
            final_str += "".join(str(x) for x in arr) + "\n"
       
        return final_str
    

state = CheckerBoard()
# print(state.as_tensor())
print(state)