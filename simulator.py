
import numpy as np
import torch
from models import device

def piece_distance(x1, y1, x2, y2):
    return int(np.sqrt((x2 - x1) ** 2 + (y2 - y1)**2))

class CheckerBoard:
    def __init__(self):
        self.board_dim = 10
        self.board = [[(i+j + 1) % 2 for i in range(self.board_dim)] for j in range(self.board_dim)]
        self.blacks = np.array([[1 if j <= 3 and self.board[i][j] == 0 else 0 for i in range(self.board_dim)] for j in range(self.board_dim)])
        self.whites = np.array([[1 if j >= (self.board_dim - 4) and self.board[i][j] == 0 else 0 for i in range(self.board_dim)] for j in range(self.board_dim)])

    def as_tensor(self):
        whites_tensor = torch.from_numpy(self.whites).unsqueeze(0)
        blacks_tensor = torch.from_numpy(self.blacks).unsqueeze(0)
        total_state = torch.cat([whites_tensor, blacks_tensor], dim=0).to(device).float()
        return total_state.unsqueeze(0)


    def check_bounds(self,x, y):
        if (x < 0 or x >= self.board_dim) or (y < 0 or y >= self.board_dim):
            return False
        return True

    def check_valid_move(self, player_board, opponent_board, x, y):
        return self.check_bounds(x, y) and opponent_board[x][y] == 0 and player_board[x][y] == 0

    def flattened_action_to_tesseract(self, action_idx):
        board_dim_sq = self.board_dim ** 2
        src_term = action_idx // board_dim_sq 
        dest_term = action_idx % board_dim_sq

        i = src_term // self.board_dim
        j = src_term % self.board_dim

        x = dest_term // self.board_dim
        y = dest_term % self.board_dim

        return (i,j), (x,y)


    def get_valid_actions(self, player):
        player_board = self.whites if player == "white" else self.blacks
        opponent_board = self.whites if player == "black" else self.blacks
        
        valid_actions = []
        for i in range(self.board_dim):
            for j in range(self.board_dim):
                action_base_idx = (i * self.board_dim + j) * (self.board_dim ** 2)
                if player_board[i][j] == 0:
                    continue
                simple_move_positions = [(i-1, j+1), (i-1, j-1)] if player == "white" else [(i+1, j+1), (i+1, j-1)] 
                simple_move_positions = filter(lambda pos: self.check_valid_move(player_board, opponent_board, pos[0], pos[1]), simple_move_positions)
                capture_move_positions = [(i+2, j+2), (i-2, j+2), (i+2, j-2), (i-2, j-2)]
                capture_move_positions = filter(lambda pos: self.check_valid_move(player_board, opponent_board, pos[0], pos[1]), capture_move_positions)
 
                for simple_position in simple_move_positions:
                    x,y = simple_position
                    src_board = np.zeros_like(player_board)
                    dest_board = np.zeros_like(player_board)
                    src_board[i][j] = 1.0
                    dest_board[x][y] = 1.0
                    action_flattened_idx = action_base_idx + (x * self.board_dim + y)
                    action_cat = np.concatenate([src_board, dest_board])
                    valid_actions.append((action_flattened_idx, action_cat))
                
                for capture_position in capture_move_positions:
                    x,y = capture_position
                    mid_x = (i+x) // 2
                    mid_y = (j+y) // 2
                    if opponent_board[mid_x][mid_y] == 1.0:
                        src_board = np.zeros_like(player_board)
                        dest_board = np.zeros_like(player_board)
                        src_board[i][j] = 1.0
                        dest_board[x][y] = 1.0
                        action_flattened_idx = action_base_idx + (x * self.board_dim + y)
                        action_cat = np.concatenate([src_board, dest_board])
                        valid_actions.append((action_flattened_idx, action_cat))

        return zip(*valid_actions)

    def execute(self, action : torch.Tensor, player):
        action_arr = action.cpu().numpy()
        src_board = action_arr[0]
        dest_board = action_arr[1]

        src_x,src_y = np.unravel_index(src_board.argmax(), src_board.shape) # (i, j)
        dest_x, dest_y = np.unravel_index(dest_board.argmax(), dest_board.shape) # (i, j) 
        player_board = self.whites if player == "white" else self.blacks
        opponent_board = self.whites if player == "black" else self.blacks

        dist = piece_distance(src_x, src_y, dest_x, dest_y)
        if dist == 1:
            player_board[src_x][src_y] = 0.0
            player_board[dest_x][dest_y] = 1.0
        if dist == 2:
            mid_x = (src_x + dest_x) // 2
            mid_y = (src_y + dest_y) // 2
            opponent_board[mid_x][mid_y] = 0.0
            player_board[src_x][src_y] = 0.0
            player_board[dest_x][dest_y] = 1.0
            
        
        self.whites = player_board if player == "white" else opponent_board
        self.blacks = opponent_board if player == "white" else player_board 
        

    
    def game_over(self):
        if np.all(self.whites == 0) or np.all(self.blacks == 0):
            return True 
        return False 
    
    def who_won(self):
        if np.all(self.whites == 0):
            return "black"
        if np.all(self.blacks == 0):
            return "white"
        return None

    def assertion_tests(self):
        assert len(state.get_valid_actions("white")) == 9 and len(state.get_valid_actions("black")) == 9

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
state.assertion_tests()