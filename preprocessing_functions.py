# Packages
import chess
import chess.engine
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import re
import os
from tqdm import tqdm
from time import sleep
import gc
from IPython.display import display, HTML, clear_output
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# Load data from Kaggle
# Data is games played by players on LiChess
raw_chess_data = pd.read_csv('chess_games.csv', usecols=['AN', 'WhiteElo'])
def select_data(lower_elo, upper_elo, train_len, test_len):
    # Only select games of certain ELO
    chess_data = raw_chess_data[(raw_chess_data['WhiteElo'] >= lower_elo) & (raw_chess_data['WhiteElo'] <= upper_elo)]
    chess_data = chess_data[['AN']] # AN is the column containing moves -> What the CNN needs for training
    chess_data = chess_data[~chess_data['AN'].str.contains('{')] # Remove 'odd' characters
    chess_data = chess_data[chess_data['AN'].str.len() > 20] # Remove games shorter than 20 moves
    chess_data_train = chess_data[0:train_len] # Select training game matches
    chess_data_test = chess_data[train_len:test_len] # Select test game matches
    return chess_data_train, chess_data_test
  
del raw_chess_data # Delete big file from memory when we don't need it anymore
gc.collect()

# Convert letters to number and vice versa
letter_2_num = {'a':0, 'b':1, 'c':2, 'd':3, 'e':4, 'f':5, 'g':6, 'h':7}
num_2_letter = {0:'a', 1:'b', 2:'c', 3:'d', 4:'e', 5:'f', 6:'g', 7:'h'}


def create_rep_layer(board, type):
    """
    Create a numerical representation of a specific type of chess piece on the chess board.

    Parameters:
        board (chess.Board): The current state of the chess board.
        type (str): The type of chess piece to represent ('p' for pawn, 'r' for rook, etc.).

    Returns:
        np.ndarray: A numpy array containing the numerical representation of the 
                    specified type of chess piece on the board. 
                    Each element in the array represents a square on the chessboard, 
                    with the specified type of piece represented as -1, the opponent's type of 
                    piece represented as 1, and empty squares represented as 0.
    """
    s = str(board)
    s = re.sub(f'[^{type}{type.upper()} \n]', '.', s)
    s = re.sub(f'{type}', '-1', s)
    s = re.sub(f'{type.upper()}', '1', s)
    s = re.sub(f'\.', '0', s)
    
    board_mat = []
    for row in s.split('\n'):
        row = row.split(' ')
        row = [int(x) for x in row]
        board_mat.append(row)
    return np.array(board_mat)


def board_2_rep(board):
    """
    Convert a chess board representation to a multi-layered numerical format 
    suitable for neural network input.

    Parameters:
        board (chess.Board): The current state of the chess board.

    Returns:
        np.ndarray: A multi-layered numpy array containing the numerical representation of 
                    the chess board. Each layer in the array represents a different type of 
                    chess piece ('p' for pawns, 'r' for rooks, etc.). 
    """
    pieces = ['p', 'r', 'n', 'b', 'q', 'k']
    layers = []
    for piece in pieces:
        layers.append(create_rep_layer(board, piece))
    board_rep = np.stack(layers)
    return board_rep


def move_2_rep(move, board):
    '''
    Convert a chess move into a representation suitable for neural network input.
        move (str): A string representing a chess move in algebraic notation, e.g., "e2e4".
        board (chess.Board): The current state of the chess board.
    Returns:
        np.ndarray: A numpy array containing the representation of the move. 
                    The array has shape (2, 8, 8)
    '''
    board.push_san(move).uci()
    move = str(board.pop())
    
    from_output_layer = np.zeros((8,8))
    from_row = 8 - int(move[1]) # 8x8 board, so 8-row. Upperleft is (8,8), but our matrix is (0,0)
    from_column = letter_2_num[move[0]]
    from_output_layer[from_row, from_column] = 1
    
    to_output_layer = np.zeros((8,8))
    to_row = 8 - int(move[3])
    to_column = letter_2_num[move[2]]
    to_output_layer[to_row, to_column] = 1
    
    return np.stack([from_output_layer, to_output_layer])


def create_move_list(s):
    return re.sub('\d*\. ','',s).split(' ')[:-1]





















