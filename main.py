from utils import *
from CNN_training import *
from evaluation import *
from minimax import *


# Load stockfish engine and configure defeault setting
install_path = 'stockfish\\stockfish-windows-x86-64-avx2.exe'
engine = chess.engine.SimpleEngine.popen_uci(install_path)
engine.configure({"UCI_LimitStrength": True,  # Limit strength allows for modifications
                  "UCI_Elo": 1320})  # Change Elo ranking

# Load CNN model weights from "chessCNN"
model = chessCNN()
if torch.cuda.is_available() == True: # If GPU is available
    model.load_state_dict(torch.load("chessCNN_weights.pt"))
    print("Weights succesfully loaded!")
else: # If only CPU is available
    model.load_state_dict(torch.load("chessCNN_weights.pt", map_location=torch.device('cpu')))
    print("Weights succesfully loaded!")


# Assign my AI to play White
global ai_white 
ai_white = True

# Match: AI vs Stockfish
# Initialize metrics to track multiple games
material_evals, piece_square_evals, number_of_moves_list, results, white_wins, black_wins, draws = init_multiple_games()
depth = 3 # Depth of minimax function
num_games = 1 # Number of games to play
for game in tqdm(range(num_games)):
    # Reset game stats
    board, number_of_moves, material_eval, piece_square_eval = init_game()
    while True: # Play until someone wins or draws
        number_of_moves += 1 # Update number of moves in game
      
        AI_move(depth, board, 5) # Perform AI move (from top 5 most probable moves)
        display_and_save_board(board, "images/image_move_" + str(game) +"_" + str(number_of_moves) + "_white.svg") # Save img
        fen = board.fen() # Convert board to 'fen' notation
        material_eval.append(eval_position(fen)) # and evaluate board
        piece_square_eval.append(evaluate_board(board))
        if board.is_game_over(): # Check if game is over when it is White's turn and track defined scores
            game_over, results, piece_square_evals, material_evals, number_of_moves_list = handle_game_over(board, results, piece_square_evals, 
                                                                                                            material_evals, number_of_moves_list, 
                                                                                                            number_of_moves, piece_square_eval, material_eval)
            break

        if board.turn == chess.BLACK: # Black's Turn
            stockfish_move(board) # Stockfish's move
            display_and_save_board(board, "images/image_move_" + str(game) +"_" + str(number_of_moves) + "_black.svg") # Save img
            fen = board.fen() # Convert board to 'fen' notation
            material_eval.append(eval_position(fen)) # and evaluate board
            piece_square_eval.append(evaluate_board(board))
        if board.is_game_over(): # Check if game is over when it's Black's turn and track defined scores
            game_over, results, piece_square_evals, material_evals, number_of_moves_list = handle_game_over(board, results, piece_square_evals, 
                                                                                                            material_evals, number_of_moves_list,
                                                                                                            number_of_moves, piece_square_eval, material_eval)
            break

# Call function to save all the scores as a DataFrame
save_dataframe(piece_square_evals, material_evals, number_of_moves_list, results, 'game_data_depth_' + str(depth))
# Use this information to investigate the AI's flaws.
# Check the images to see how to struggles to win even when it is far a head in material and piece-square score.
