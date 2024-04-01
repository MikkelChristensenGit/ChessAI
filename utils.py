# Helper functions
# Check if it possible to checkmate
def check_mate_single(board):
    board = board.copy()
    legal_moves = list(board.legal_moves)
    for move in legal_moves:
        board.push_uci(str(move))
        if board.is_checkmate():
            move = board.pop()
            return move
        _ = board.pop()
        
# Make moves more or less random
def distribution_over_moves(vals):
    probs = np.array(vals)
    #probs = np.exp(probs) # Change the prob. distribution
    #probs = probs / probs.sum()
    #probs = probs ** 3
    probs = probs / probs.sum()
    return probs

# Choose a move
# Select 'moves_consider' amount of moves from the CNN output
#    Minimax will check all the possible paths for these moves
def choose_move(board, player, color, moves_consider):
    legal_moves = list(board.legal_moves)
    move = check_mate_single(board)
    if move is not None:
        return move
    
    x = torch.Tensor(board_2_rep(board))
    if color == chess.BLACK:
        x *= -1
    x = x.unsqueeze(0)
    move = model(x)

    vals = []
    froms = [str(legal_move)[:2] for legal_move in legal_moves]
    froms = list(set(froms))

    for from_ in froms:
        val = move[:,0,:][0, 8 - int(from_[1]), letter_2_num[from_[0]]]
        vals.append(val.detach().cpu().numpy())
    probs = distribution_over_moves(vals)
    froms_arr = np.array([froms, probs]).T
    froms_arr = froms_arr[froms_arr[:, 1].argsort()]
    froms_list = froms_arr[-moves_consider:][:,0]
    
    vals = []
    vals_move_to = []
    from_list = []
    for legal_move in legal_moves:
        from_ = str(legal_move)[:2]
        for choosen_from in froms_list:
            if from_ == choosen_from:
                to = str(legal_move)[2:]
                val = move[:,1,:][0, 8 - int(to[1]), letter_2_num[to[0]]]
                from_list.append(from_)
                vals.append(val.detach().cpu().numpy())  ##
                vals_move_to.append(to)
                
    probs_to = distribution_over_moves(vals)
    to_arr = np.array([from_list, vals_move_to, probs_to]).T
    to_arr = to_arr[to_arr[:, 2].argsort()]
    best_moves = to_arr[-moves_consider:] 
    top_moves = []
    for moves in range(len(best_moves)):
        choosen_move = best_moves[moves,0] + best_moves[moves,1]
        top_moves.append(choosen_move)
    return top_moves

def save_svg(svg, filepath):
    """Save svg content in filepath
    :param str  svg:        SVG content
    :param str  filepath:   Path of the SVG file to save
    """
    try:
        file_handle = open(filepath, 'w')
    except IOError as e:
        print(str(e))
        exit(1)

    file_handle.write(svg)
    file_handle.close() 

def display_board(board, use_svg):
    if use_svg:
        return board._repr_svg_()
    else:
        return "<pre>" + str(board) + "</pre>"

# Initialize counters and metrics to track when looping over multiple games
def init_multiple_games():
    material_evals = []
    piece_square_evals = []
    number_of_moves_list = []
    results = []
    white_wins = 0
    black_wins = 0
    draws = 0
    return material_evals, piece_square_evals, number_of_moves_list, results, white_wins, black_wins, draws

def init_game():
    board = chess.Board()
    number_of_moves = 0
    material_eval = []
    piece_square_eval = []
    return board, number_of_moves, material_eval, piece_square_eval

def AI_move(minimax_depth, board, moves_consider):
    next_move_white = minimax_root(minimax_depth, board, moves_consider)
    move_to_white = board.san(next_move_white)
    board.push_san(move_to_white)

def stockfish_move(board):
    stockfish_move = engine.play(board, chess.engine.Limit(time=0.1))
    board.push(stockfish_move.move)

def handle_game_over(board, results, piece_square_evals, material_evals, number_of_moves_list, number_of_moves, piece_square_eval, material_eval):
    """Handle game over conditions and append metrics."""
    if board.result() == '1-0':
        results.append(1)
        print('White Wins!')
    elif board.result() == '0-1':
        results.append(-1)
        print('Black Wins!')
    else:
        results.append(0)
        print('Draw!')

    piece_square_evals.append(piece_square_eval)
    material_evals.append(material_eval)
    number_of_moves_list.append(number_of_moves)
    return True, results, piece_square_evals, material_evals, number_of_moves_list
  
def display_and_save_board(board, filename):
    """Display the chess board and save it as an SVG image."""
    html = display_board(board, True)
    save_svg(html, filename)
    clear_output(wait=True)
    display(HTML(html))


def save_dataframe(piece_square_evals, material_evals, number_of_moves_list, results, prefix):
    """Convert arrays to a DataFrame and save it."""
    data = {
        'piece_square_evals': piece_square_evals,
        'material_evals': material_evals,
        'number_of_moves_list': number_of_moves_list,
        'results': results
    }
    df = pd.DataFrame(data)
    df.to_csv(f'{prefix}.csv', index=False)
