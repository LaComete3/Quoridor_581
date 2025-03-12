# Quoridor_581

## Overview
Quoridor_581 is an implementation of the Quoridor board game. This project includes the game logic, environment setup for reinforcement learning, and various agents to play the game.

![Quoridor Board Game](ressources/Gigamic-Quoridor-Mini.jpg)

## Features
- Full implementation of Quoridor game rules.
- 2 players and 4 players ???
- Environment setup compatible with OpenAI Gym ???
- Visualization tools to render the game state.
- Agents including random, minimax, DQN-based and MCTS-based agents for playing the game.


## Usage
### Running the Game
You can run the game using the provided Jupyter notebooks.

#### Using Jupyter Notebooks
1. Open `main.ipynb`
2. Follow the description and instructions of the Notebook


### Training Agents
You can train reinforcement learning agents using the `qoridor/DQN.ipynb` notebook. This notebook sets up the environment and trains a DQN agent to play Quoridor.

## Detailed Project Structure
- `main.ipynb`: Jupyter notebook to run and visualize the game.
- `README.md`: Project documentation.
- `qoridor/`:
    - `game.py`: Description TODO
    - `environment.py`: Description TODO
    - `board.py`: Description TODO
    - `move.py`: Description TODO
    - `rules.py`: Description TODO
    - `state`: Description TODO
    - `visualization`: Description TODO

- `agents/`:
    - `random_agent.py`: Implementation of an agent that makes random moves.
    - `minimax.py`: Implementation of an agent following minimax. TO BE DONE
    - `DQN.ipynb`: Jupyter notebook to train a DQN agent. TO BE DONE



