# Chess AI using CNN and Minimax Algorithm

This Python script provides an implementation of a chess AI that utilizes Convolutional Neural Networks (CNN) for move prediction and the minimax algorithm for decision-making. The AI is designed to play chess against human players or other AIs.

## Overview
The script consists of several Python files and helper functions organized into separate folders. Here's a brief overview of each component:


## Folder Structure

- **utils.py**: Contains utility functions required for data preprocessing, board representation, move conversion, game handling, and visualization.
- **CNN_training.py**: Script for training the CNN model using the provided dataset. It includes data preprocessing, dataset creation, model definition, training loop, and model saving.
- **evaluation.py**: Evaluates the trained model's performance based on material and piece-square tables.
- **minimax.py**: Implements the Minimax algorithm with alpha-beta pruning for decision making in the Chess AI.
- **main.py**: Initializes Stockfish (black) by loading the engine and my AI (white) by loading the CNN weights. Running this script  makes my AI and Stockfish play chess against each other. All desired match statistics are kept track of during the match and will be saved as a dataframe for further investigation.

## Usage
To use the chess AI, follow these steps:

Ensure that you have Python installed on your system along with the required libraries listed in requirements.txt.

Run the CNN_training.py script to train the CNN model. Adjust the training parameters as needed.

Once the model is trained, you can run the main script to start playing against the AI. You can choose to play against the AI or watch it play against another AI.
