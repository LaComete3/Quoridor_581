# Quoridor_581

## Overview
Quoridor_581 is an implementation of the Quoridor board game. This project includes the game logic, environment setup for reinforcement learning, and various agents to play the game.

## Features
- Full implementation of Quoridor game rules.
- Environment setup compatible with OpenAI Gym ???
- Visualization tools to render the game state.
- Agents including random, DQN-based and MCTS-based agents for playing the game.


## Usage
### Running the Game
You can run the game using the provided Jupyter notebooks.

#### Using Jupyter Notebooks
1. Open `main.ipynb` in Jupyter Notebook.
2. Follow the description and instructions of the Notebook


### Training Agents
You can train reinforcement learning agents using the `qoridor/DQN.ipynb` notebook. This notebook sets up the environment and trains a DQN agent to play Quoridor.

## Detailed Project Structure
- `main.ipynb`: Jupyter notebook to run and visualize the game.
- `README.md`: Project documentation.
- `qoridor/`:
    - `game.py`: TODO
    - `environment.py`: TODO
    - `board.py`: TODO
    - `move.py`: TODO
    - `rules.py`: TODO
    - `state`: TODO
    - `visualization`: TODO

- `agents/`:
    - `random_agent.py`: Implementation of an agent that makes random moves.
    - `minimax.py`: Implementation of an agent following minimax.
    - `DQN.ipynb`: Jupyter notebook to train a DQN agent.



