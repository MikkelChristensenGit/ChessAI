# Minimax algorithm with alpha-beta pruning
# It looks at all the possible paths for the 'moves_consider' number of moves outputted by the CNN.
# So, instead of looking all at all possible paths, we only look at the most probable, according to the CNN.
def minimax(depth, board, alpha, beta, moves_consider, is_maximising_player):
    
    if depth == 0 or board.is_game_over():
        return - evaluate_board(board)
    elif depth > 3:
        uci_moves = choose_move(board, 0, chess.WHITE, moves_consider)
        legal_moves = [chess.Move.from_uci(uci) for uci in uci_moves]
    else:
        legal_moves = list(board.legal_moves)

    if is_maximising_player:
        best_move = -9999
        for move in legal_moves:
            board.push(move)
            best_move = max(best_move, minimax(depth-1, board, alpha, beta, moves_consider, not is_maximising_player))
            board.pop()
            alpha = max(alpha, best_move)
            if beta <= alpha:
                return best_move
        return best_move
    else:
        best_move = 9999
        for move in legal_moves:
            board.push(move)
            best_move = min(best_move, minimax(depth-1, board, alpha, beta, moves_consider, not is_maximising_player))
            board.pop()
            beta = min(beta, best_move)
            if beta <= alpha:
                return best_move
        return best_move
    

def minimax_root(depth, board, moves_consider, is_maximising_player=True):
    # only search the top 50% moves
    legal_moves = choose_move(board, 0, chess.WHITE, moves_consider)
    best_move = -9999
    best_move_found = None
    try: # This is the standard routine
        for move in legal_moves:
            move = chess.Move.from_uci(move)
            board.push(move)
            value = minimax(depth - 1, board, -10000, 10000, moves_consider, not is_maximising_player)
            board.pop()
            if value >= best_move:
                best_move = value
                best_move_found = move
        return best_move_found
    except TypeError: # When chess engine only has one available move, we can't iterate
        return legal_moves
