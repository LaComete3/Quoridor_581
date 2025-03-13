from typing import Tuple, Optional
from copy import deepcopy
from qoridor.game import QoridorGame
from qoridor.move import Move, MoveType
from qoridor.board import WallOrientation
from qoridor.visualization import QoridorVisualizer
from qoridor.board import Board

class MinimaxAgent:
    def __init__(self, depth: int):
        self.depth = depth

    def minimax(self, game: QoridorGame, depth: int, maximizing_player: bool) -> Tuple[int, Optional[Move]]:
        if depth == 0 or game.is_game_over():
            return self.evaluate(game), None

        legal_moves = game.get_legal_moves()
        best_move = None

        if maximizing_player:
            max_eval = float('-inf')
            for move in legal_moves:
                game_copy = deepcopy(game)
                game_copy.make_move(move)
                eval, _ = self.minimax(game_copy, depth - 1, False)
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
            return max_eval, best_move
        else:
            min_eval = float('inf')
            for move in legal_moves:
                game_copy = deepcopy(game)
                game_copy.make_move(move)
                eval, _ = self.minimax(game_copy, depth - 1, True)
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
            return min_eval, best_move

    def evaluate(self, game: QoridorGame) -> int:
        """
        Evaluate the game state for the current player.
        
        Args:
            game: The current game state
            
        Returns:
            An integer representing the evaluation score
        """
        player1_pos = game.state.board.get_player_position(1)
        player2_pos = game.state.board.get_player_position(2)
        
        # Distance to goal (rows)
        player1_distance = game.state.board.size - 1 - player1_pos[0]
        player2_distance = player2_pos[0]
        
        # Number of walls remaining
        player1_walls = game.get_walls_left(1)
        player2_walls = game.get_walls_left(2)
        
        # Evaluation score
        score = (player2_distance - player1_distance) + (player1_walls - player2_walls)
        
        return score

    def get_move(self, game: QoridorGame) -> Move:
        _, best_move = self.minimax(game, self.depth, True)
        return best_move