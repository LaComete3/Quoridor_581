from qoridor.game import QoridorGame
from qoridor.move import Move, MoveType
from qoridor.board import WallOrientation

game = QoridorGame(board_size=5, num_walls=2)
game.reset()
