import random
from qoridor.game import QoridorGame
from qoridor.move import Move

class RandomAgent:
    def __init__(self):
        pass

    def get_move(self, game: QoridorGame) -> Move:
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            raise ValueError("No legal moves available")
        return random.choice(legal_moves)

