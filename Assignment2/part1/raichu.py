#
# raichu.py : Play the game of Raichu
#
# Code by: Harsh Srivastava <hsrivas>
#        : Ritwik Budhiraja <rbudhira>
#        : Yash Kalpesh Shah <yashah>
#
# Based on skeleton code by D. Crandall, Oct 2021
#

import json
import math
import os
import sys
import signal
import random
import time

# Just flip DEBUG to True for enabling debug output
DEBUG = False

# Custom printd for debug
def printd(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

# Create a custom made MutableString class
class MutableString:
    def __init__(self, string):
        self.__string__ = string

    def __getitem__(self, key):
        return self.__string__[key]

    def __setitem__(self, key, value):
        self.__string__ = self.__string__[0:key] + value + self.__string__[key + 1:]

    def __str__(self):
        return self.__string__

    def __len__(self):
        return len(self.__string__)

    def count(self, *args):
        return self.__string__.count(*args)

def board_to_string(board, N):
    return "\n".join(board[i:i+N] for i in range(0, len(board), N))

def convert_index_to_rc(N, i):
    return (i // N, i % N)

def convert_rc_to_index(N, r, c):
    if r not in range(0, N) or c not in range(0, N):
        return None
    return r * N + c

def is_last_row_index(rc_index, N, player):
    if player == "w":
        return (rc_index // N) == (N - 1)
    else:
        return (rc_index // N) == 0

def is_opposing_piece(piece: str, target_piece: str):
    return piece.lower() != target_piece.lower() and piece != "." and target_piece != "."

def is_friendly_piece(piece: str, target_piece: str):
    return (piece.lower() == target_piece.lower() or piece == target_piece) \
        and piece != "." and target_piece != "."

def get_pichu_moves(board: str, N: int, pieceIndex: int, piece: str, player: str, player_pieces: str):
    r, c = convert_index_to_rc(N, pieceIndex)

    invert_up_down_factor = 1
    if player == "b":
        invert_up_down_factor = -1

    moves = []

    # Calculate move indices
    move_rd_cl = convert_rc_to_index(N, r + 1 * invert_up_down_factor, c - 1)
    move_rd_cr = convert_rc_to_index(N, r + 1 * invert_up_down_factor, c + 1)
    move_rdd_cll = convert_rc_to_index(N, r + 2 * invert_up_down_factor, c - 2)
    move_rdd_crr = convert_rc_to_index(N, r + 2 * invert_up_down_factor, c + 2)

    if move_rdd_cll and board[move_rdd_cll] == "." and is_opposing_piece(piece, board[move_rd_cl]) and str.islower(board[move_rd_cl]):
        new_board = MutableString(board)
        new_board[move_rdd_cll] = player_pieces[-1] \
                                if is_last_row_index(move_rdd_cll, N, player) \
                                else piece
        new_board[move_rd_cl] = "."
        new_board[pieceIndex] = "."
        moves.append({"new_board": str(new_board), "gain": 1})

    if move_rdd_crr and board[move_rdd_crr] == "." and is_opposing_piece(piece, board[move_rd_cr]) and str.islower(board[move_rd_cr]):
        new_board = MutableString(board)
        new_board[move_rdd_crr] = player_pieces[-1] \
                                if is_last_row_index(move_rdd_crr, N, player) \
                                else piece
        new_board[move_rd_cr] = "."
        new_board[pieceIndex] = "."
        moves.append({"new_board": str(new_board), "gain": 1})

    if move_rd_cl and board[move_rd_cl] == ".":
        new_board = MutableString(board)
        new_board[move_rd_cl] = player_pieces[-1] \
                                if is_last_row_index(move_rd_cl, N, player) \
                                else piece
        new_board[pieceIndex] = "."
        moves.append({"new_board": str(new_board), "gain": 0})

    if move_rd_cr and board[move_rd_cr] == ".":
        new_board = MutableString(board)
        new_board[move_rd_cr] = player_pieces[-1] \
                                if is_last_row_index(move_rd_cr, N, player) \
                                else piece
        new_board[pieceIndex] = "."
        moves.append({"new_board": str(new_board), "gain": 0})

    return moves

def get_pikachu_moves(board: str, N: int, pieceIndex: int, piece: str, player: str, player_pieces: str):
    r, c = convert_index_to_rc(N, pieceIndex)

    invert_up_down_factor = 1
    if player == "b":
        invert_up_down_factor = -1

    moves = []

    move_rd = convert_rc_to_index(N, r + (1 * invert_up_down_factor), c)
    move_rdd = convert_rc_to_index(N, r + (2 * invert_up_down_factor), c)
    move_rddd = convert_rc_to_index(N, r + (3 * invert_up_down_factor), c)
    move_cl = convert_rc_to_index(N, r, c - 1)
    move_cll = convert_rc_to_index(N, r, c - 2)
    move_clll = convert_rc_to_index(N, r, c - 3)
    move_cr = convert_rc_to_index(N, r, c + 1)
    move_crr = convert_rc_to_index(N, r, c + 2)
    move_crrr = convert_rc_to_index(N, r, c + 3)

    # 3 Square Forward
    if move_rddd != None and board[move_rddd] == "." and board[move_rdd] == "." and is_opposing_piece(piece, board[move_rd]) and board[move_rd] not in "@$":
        new_board = MutableString(board)
        new_board[move_rddd] = player_pieces[-1] \
                                if (invert_up_down_factor == 1 and is_last_row_index(move_rddd, N, player)) \
                                else piece
        new_board[move_rd] = "."
        new_board[pieceIndex] = "."
        moves.append({ "new_board": str(new_board), "gain": 1 })
    elif move_rddd != None and board[move_rddd] == "." and board[move_rd] == "." and is_opposing_piece(piece, board[move_rdd]) and board[move_rdd] not in "@$":
        new_board = MutableString(board)
        new_board[move_rddd] = player_pieces[-1] \
                                if (invert_up_down_factor == 1 and is_last_row_index(move_rddd, N, player)) \
                                else piece
        new_board[move_rdd] = "."
        new_board[pieceIndex] = "."
        moves.append({ "new_board": str(new_board), "gain": 1 })

    # 2 Square Forward, if opposing piece, then remove it
    if move_rdd != None and board[move_rdd] == ".":
        new_board = MutableString(board)
        new_board[move_rdd] = player_pieces[-1] \
                                if (invert_up_down_factor == 1 and is_last_row_index(move_rdd, N, player)) \
                                else piece
        gain = 0
        if is_opposing_piece(piece, board[move_rd]):
            new_board[move_rd] = "."
            gain = 1
        elif not is_friendly_piece(piece, board[move_rd]):
            new_board[pieceIndex] = "."
            moves.append({ "new_board": str(new_board), "gain": gain })

    # 1 Square Forward
    if move_rd != None and board[move_rd] == ".":
        new_board = MutableString(board)
        new_board[move_rd] = player_pieces[-1] \
                                if (invert_up_down_factor == 1 and is_last_row_index(move_rd, N, player)) \
                                else piece
        new_board[pieceIndex] = "."
        moves.append({ "new_board": str(new_board), "gain": 0 })

    # 3 Square Left
    if move_clll != None and board[move_clll] == "." and board[move_cll] == "." and is_opposing_piece(piece, board[move_cl]):
        new_board = MutableString(board)
        new_board[move_clll] = piece
        new_board[move_cl] = "."
        new_board[pieceIndex] = "."
        moves.append({ "new_board": str(new_board), "gain": 1 })
    elif move_clll != None and board[move_clll] == "." and board[move_cl] == "." and is_opposing_piece(piece, board[move_cll]):
        new_board = MutableString(board)
        new_board[move_clll] = piece
        new_board[move_cll] = "."
        new_board[pieceIndex] = "."
        moves.append({ "new_board": str(new_board), "gain": 1 })

    # 2 Square Left, if opposing piece, then remove it
    if move_cll != None and board[move_cll] == ".":
        new_board = MutableString(board)
        new_board[move_cll] = piece
        gain = 0
        if is_opposing_piece(piece, board[move_cl]):
            new_board[move_cl] = "."
            gain = 1
        new_board[pieceIndex] = "."
        moves.append({ "new_board": str(new_board), "gain": gain })

    # 1 Square Left
    if move_cl != None and board[move_cl] == ".":
        new_board = MutableString(board)
        new_board[move_cl] = piece
        new_board[pieceIndex] = "."
        moves.append({ "new_board": str(new_board), "gain": 0 })

    # 3 Square Right
    if move_crrr != None and board[move_crrr] == "." and board[move_crr] == "." and is_opposing_piece(piece, board[move_cr]):
        new_board = MutableString(board)
        new_board[move_crrr] = piece
        new_board[move_cr] = "."
        new_board[pieceIndex] = "."
        moves.append({ "new_board": str(new_board), "gain": 1 })
    elif move_crrr != None and board[move_crrr] == "." and board[move_cr] == "." and is_opposing_piece(piece, board[move_crr]):
        new_board = MutableString(board)
        new_board[move_crrr] = piece
        new_board[move_crr] = "."
        new_board[pieceIndex] = "."
        moves.append({ "new_board": str(new_board), "gain": 1 })

    # 2 Square Right, if opposing piece, then remove it
    if move_crr != None and board[move_crr] == ".":
        new_board = MutableString(board)
        new_board[move_crr] = piece
        gain = 0
        if is_opposing_piece(piece, board[move_cr]):
            new_board[move_cr] = "."
            gain = 1
        new_board[pieceIndex] = "."
        moves.append({ "new_board": str(new_board), "gain": gain })

    # 1 Square Right
    if move_cr != None and board[move_cr] == ".":
        new_board = MutableString(board)
        new_board[move_cr] = piece
        new_board[pieceIndex] = "."
        moves.append({ "new_board": str(new_board), "gain": 0 })

    return moves

def get_raichu_moves(board: str, N: int, pieceIndex: int, piece: str, player: str, player_pieces: str):
    r, c = convert_index_to_rc(N, pieceIndex)

    moves = []

    #Move forward rows
    for delta in range(r + 1, N):
        from_c_to_delta = ""
        for delta_r in range(r + 1, delta):
            from_c_to_delta += board[convert_rc_to_index(N, delta_r, c)]

        from_c_to_delta_without_blank = from_c_to_delta.replace(".", "")
        if len(from_c_to_delta_without_blank) == 0:
            move_rdn = convert_rc_to_index(N, delta, c)
            if move_rdn != None and board[move_rdn] == ".":
                new_board = MutableString(board)
                new_board[move_rdn] = piece
                new_board[pieceIndex] = "."
                moves.append({ "new_board": str(new_board), "gain": calculate_cost(new_board, player) })
        elif len(from_c_to_delta_without_blank) == 1 and from_c_to_delta_without_blank in get_opponent_pieces(player):
            move_rdn = convert_rc_to_index(N, delta, c)
            if move_rdn != None and board[move_rdn] == ".":
                new_board = MutableString(board)
                for delta_r in range(r + 1, delta):
                    new_board[convert_rc_to_index(N, delta_r, c)] = "."
                new_board[move_rdn] = piece
                new_board[pieceIndex] = "."
                moves.append({ "new_board": str(new_board), "gain": calculate_cost(new_board, player) })

    # Move backward rows
    for delta in range(r - 1, -1, -1): # Reverse range for 'range(0, r)'
        from_c_to_delta = ""
        for delta_r in range(r - 1, delta, -1):
            from_c_to_delta += board[convert_rc_to_index(N, delta_r, c)]

        from_c_to_delta_without_blank = from_c_to_delta.replace(".", "")
        if len(from_c_to_delta_without_blank) == 0:
            move_rdn = convert_rc_to_index(N, delta, c)
            if move_rdn != None and board[move_rdn] == ".":
                new_board = MutableString(board)
                new_board[move_rdn] = piece
                new_board[pieceIndex] = "."
                # print("Back0:", new_board)
                moves.append({ "new_board": str(new_board), "gain": calculate_cost(new_board, player) })
        elif len(from_c_to_delta_without_blank) == 1 and from_c_to_delta_without_blank in get_opponent_pieces(player):
            move_rdn = convert_rc_to_index(N, delta, c)
            if move_rdn != None and board[move_rdn] == ".":
                new_board = MutableString(board)
                for delta_r in range(r - 1, delta, -1):
                    new_board[convert_rc_to_index(N, delta_r, c)] = "."
                new_board[move_rdn] = piece
                new_board[pieceIndex] = "."
                moves.append({ "new_board": str(new_board), "gain": calculate_cost(new_board, player) })

    # Move right columns
    for delta in range(c + 1, N):
        from_c_to_delta = ""
        for delta_c in range(c + 1, delta):
            from_c_to_delta += board[convert_rc_to_index(N, r, delta_c)]

        from_c_to_delta_without_blank = from_c_to_delta.replace(".", "")
        if len(from_c_to_delta_without_blank) == 0:
            move_cdn = convert_rc_to_index(N, r, delta)
            if move_cdn != None and board[move_cdn] == ".":
                new_board = MutableString(board)
                new_board[move_cdn] = piece
                new_board[pieceIndex] = "."
                moves.append({ "new_board": str(new_board), "gain": calculate_cost(new_board, player) })
        elif len(from_c_to_delta_without_blank) == 1 and from_c_to_delta_without_blank in get_opponent_pieces(player):
            move_cdn = convert_rc_to_index(N, r, delta)
            if move_cdn != None and board[move_cdn] == ".":
                new_board = MutableString(board)
                for delta_c in range(c + 1, delta):
                    new_board[convert_rc_to_index(N, r, delta_c)] = "."
                new_board[move_cdn] = piece
                new_board[pieceIndex] = "."
                moves.append({ "new_board": str(new_board), "gain": calculate_cost(new_board, player) })

    # Move left columns
    for delta in range(c - 1, -1, -1): # Reverse range for 'range(0, r)'
        from_c_to_delta = "".join([board[convert_rc_to_index(N, r, delta_c)] for delta_c in range(delta, -1, -1)])
        from_c_to_delta_without_blank = from_c_to_delta.replace(".", "")
        if len(from_c_to_delta_without_blank) == 0:
            move_cdn = convert_rc_to_index(N, r, delta)
            if move_cdn != None and board[move_cdn] == ".":
                new_board = MutableString(board)
                new_board[move_cdn] = piece
                new_board[pieceIndex] = "."
                moves.append({ "new_board": str(new_board), "gain": calculate_cost(new_board, player) })
        elif len(from_c_to_delta_without_blank) == 1 and from_c_to_delta_without_blank in get_opponent_pieces(player):
            move_cdn = convert_rc_to_index(N, r, delta)
            if move_cdn != None and board[move_cdn] == ".":
                new_board = MutableString(board)
                for delta_c in range(delta, c):
                    new_board[convert_rc_to_index(N, r, delta_c)] = "."
                new_board[move_cdn] = piece
                new_board[pieceIndex] = "."
                moves.append({ "new_board": str(new_board), "gain": calculate_cost(new_board, player) })

    for direction in zip([-1, -1, 1, 1], [-1, 1, -1, 1]):
        printd("Direction:", direction)
        for target_r, target_c in zip(range(r + direction[0], N if direction[0] == 1 else -1, direction[0]), \
                                      range(c + direction[1], N if direction[1] == 1 else -1, direction[1])):
            printd("    Target (r, c) =", target_r, target_c)
            move_rdn = convert_rc_to_index(N, target_r, target_c)

            # if board[move_rdn] in get_player_pieces(player):
            #     printd("        Skip")
            #     break

            from_c_to_delta = ""
            for delta_r, delta_c in zip(range(r + direction[0], target_r, direction[0]), \
                                        range(c + direction[1], target_c, direction[1])):
                from_c_to_delta += board[convert_rc_to_index(N, delta_r, delta_c)]
            printd("        Route:", from_c_to_delta)

            from_c_to_delta_without_blank = from_c_to_delta.replace(".", "")
            if len(from_c_to_delta_without_blank) == 0:
                printd("            Simply move, all spaces")
                if move_rdn != None and board[move_rdn] == ".":
                    new_board = MutableString(board)
                    new_board[move_rdn] = piece
                    new_board[pieceIndex] = "."
                    moves.append({ "new_board": str(new_board), "gain": calculate_cost(new_board, player) })
            elif len(from_c_to_delta_without_blank) == 1 and from_c_to_delta_without_blank in get_opponent_pieces(player):
                printd("            Move with cutting one opponent")
                if move_rdn != None and board[move_rdn] == ".":
                    new_board = MutableString(board)
                    printd("Row:", target_r + direction[0], N if direction[0] == 1 else -1, direction[0])
                    printd("Row:", target_c + direction[1], N if direction[1] == 1 else -1, direction[1])
                    for delta_r, delta_c in zip(range(r, target_r, direction[0]), \
                                                range(c, target_c, direction[1])):
                        printd("Clear: ", delta_r, delta_c)
                        new_board[convert_rc_to_index(N, delta_r, delta_c)] = "."
                    new_board[move_rdn] = player_pieces[-1] if move_rdn == (N - 1) else piece
                    new_board[pieceIndex] = "."
                    moves.append({ "new_board": str(new_board), "gain": calculate_cost(new_board, player) })

    return moves

def get_player_pieces(player: str):
    return "bB$" if player.lower() == "b" else "wW@"

def get_opponent_pieces(player: str):
    return "bB$" if player.lower() != "b" else "wW@"

def get_opponent(player: str):
    return "b" if player == "w" else "w"

def randomize(l: list):
    N = len(l)
    for i in range(0, N):
        random_idx = random.randint(0, N - 1)
        l[random_idx], l[i] = l[i], l[random_idx]

def successors(board: str, N: int, player: str):
    player_pieces = get_player_pieces(player)
    piece_indices = [i for i in range(len(board)) if board[i] in player_pieces]
    # printd(player_pieces, piece_indices)

    allMoves = []
    for pieceIndex in piece_indices:
        piece = board[pieceIndex]

        if piece in "@$":
            allPieceMoves = get_raichu_moves(board, N, pieceIndex, piece, player, player_pieces)

            # # Uncomment for debugging raichu's moves
            # printd("\033[45;37;1m Raichu \033[0m:" + piece)
            # for pichuMove in allMoves:
            #     printd("Move:\n" + board_to_string(pichuMove, N))

            allMoves += allPieceMoves
        elif piece in "BW":
            allPieceMoves = get_pikachu_moves(board, N, pieceIndex, piece, player, player_pieces)

            # # Uncomment for debugging pikachu's moves
            # printd("\033[44;37;1m Pikachu \033[0m:" + piece)
            # for pichuMove in allMoves:
            #     printd("Move:\n" + board_to_string(pichuMove, N))

            allMoves += allPieceMoves
        elif piece in "bw":
            allPieceMoves = get_pichu_moves(board, N, pieceIndex, piece, player, player_pieces)

            # # Uncomment for debugging pichu's moves
            # printd("\033[43;37;1m Pichu \033[0m:" + piece)
            # for pichuMove in allMoves:
            #     printd("Move:\n" + board_to_string(pichuMove, N))

            allMoves += allPieceMoves

    # Randomize order of same cost states
    randomize(allMoves)

    # Then sort them in place preserving the random order
    allMoves.sort(key=lambda x: x["gain"])

    # Form set out of it without losing sorted order
    result = []
    for move in allMoves:
        if move["new_board"] not in result:
            result.append(move["new_board"])
    return result

def count_player(board: str, player: str):
    pieces = get_player_pieces(player)
    return sum([board.count(key) for key in pieces])

def calculate_cost(board: str, player: str, aggression_level: int = 0.5, max_aggression_level: int = 1.0):
    # Calculate total value of pieces
    #   - Pichu = 1
    #   - Pikachu = 100
    #   - Raichu = 1000

    # Player pieces and points
    pieces = get_player_pieces(player)
    player_pieces_count = sum([board.count(piece) for piece in pieces])
    player_points = board.count(pieces[0]) + board.count(pieces[1]) * 100 + board.count(pieces[2]) * 1000

    # Opponent pieces and points
    opponent_pieces = get_opponent_pieces(player)
    opponent_pieces_count = sum([board.count(piece) for piece in pieces])
    opponent_points = board.count(opponent_pieces[0]) + board.count(opponent_pieces[1]) * 100 + board.count(opponent_pieces[2]) * 1000

    # Total pieces
    total_pieces = len(board) - board.count(".")
    aggression = aggression_level / max_aggression_level

    # # Tried controlling aggression through player piece count
    # player_presence = player_pieces_count / total_pieces
    # opponent_presence = opponent_pieces_count / total_pieces

    return (player_points * aggression - opponent_points * (1 - aggression))

def expand_max_tree(results_store, board, N, player, timelimit, height, hlimit, alpha, beta, aggression_level: int = 50, max_aggression_level: int = 100):
    # Initialise lookup key for taking advantage of memoization
    lookup_key = "{}-{}-{}".format(hlimit - height, player, board)

    # If lookup is found
    if lookup_key in results_store:
        return (board, results_store[lookup_key])

    printd("DEBUG:", height, board, alpha, beta)

    # Get all possible children nodes / successor moves
    successors_list = successors(board, N, player)
    if player == "b":
        successors_list.reverse()

    # Check if height limit reached or leaf nodes
    if height == hlimit or len(successors_list) == 0:
        return (board, calculate_cost(board, player))

    # Initialise to minus infinity initially
    current_max = -math.inf
    current_max_board = None
    current_alpha = alpha

    # Go through the list
    for successor in successors_list:
        # Proceed with next level MIN tree
        result = expand_min_tree(results_store, successor, N, player, timelimit, height + 1, hlimit, current_alpha, beta, aggression_level, max_aggression_level)

        # Check current max
        if current_max < result[1]:
            current_max = result[1]
            current_max_board = successor

        # Check alpha max
        current_alpha = max(current_alpha, current_max)
        
        # Check if beta is less than current alpha then break
        if beta <= current_alpha:
            break

    # Store the result to the results store for memoization
    results_store[lookup_key] = current_max

    # Return the current max
    return (current_max_board, current_max)

def expand_min_tree(results_store, board, N, player, timelimit, height, hlimit, alpha, beta, aggression_level: int = 50, max_aggression_level: int = 100):
    # Initialise lookup key for taking advantage of memoization
    lookup_key = "{}-{}-{}".format(hlimit - height, player, board)

    # If lookup is found
    if lookup_key in results_store:
        return (board, results_store[lookup_key])

    printd("DEBUG:", height, board, alpha, beta)

    # Get all possible children nodes / successor moves
    successors_list = successors(board, N, player)
    if player == "b":
        successors_list.reverse()

    # Check if height limit reached or leaf nodes
    if height == hlimit or len(successors_list) == 0:
        return (board, calculate_cost(board, player))

    # Initialise to minus infinity initially
    current_min = math.inf
    current_min_board = None
    current_beta = beta

    # Go through the list
    for successor in successors_list:
        # Proceed with next level MIN tree
        result = expand_max_tree(results_store, successor, N, player, timelimit, height + 1, hlimit, alpha, current_beta, aggression_level, max_aggression_level)

        # Check current min
        if current_min > result[1]:
            current_min = result[1]
            current_min_board = successor

        # Check beta min
        current_beta = min(current_beta, current_min)
        
        # Check if beta is less than current alpha then break
        if current_beta <= alpha:
            break

    # Store the result to the results store for memoization
    results_store[lookup_key] = current_min

    # Return the current min
    return (current_min_board, current_min)

def find_best_move(board, N, player, timelimit):
    # This sample code just returns the same board over and over again (which
    # isn't a valid move anyway.) Replace this with your code!
    #

    # Form the result store filename
    result_store_filename = "hry-raichu-results-store-" + str(N) + ".json"

    # Memoization store
    result_store = {}

    # Store received board states to increase aggression in case opponent repeats moves
    historical_moves = {}

    # Store the current time
    start_time = time.time()

    # Read memoized results store if file exists
    if os.path.exists(result_store_filename):
        # result_store = json.loads(open(result_store_filename, "r").read())
        historical_moves = json.loads(open(result_store_filename, "r").read())

    # Add the current board into historical moves
    historical_moves[str(time.time_ns())] = board
    print("PID", os.system("ps ux | grep "))

    # Signal handler code based on the method described in the below link:
    #   - https://stackoverflow.com/questions/1112343/how-do-i-capture-sigint-in-python
    def signal_handler(sig, frame):
        # We handle the process kill SIGINT signal to write the cache to disk
        # with open(result_store_filename, "w") as handle:
        #     handle.write(json.dumps(result_store, indent=4))
        # handle.close()
        with open(result_store_filename, "w") as handle:
            handle.write(json.dumps(historical_moves, indent=4))
        handle.close()
        printd("Result Store written to disk in file:", result_store_filename)
        sys.exit(0)

    # Set the signal handler function
    signal.signal(signal.SIGINT, signal_handler)
    
    # Store the current depth of tree
    #   - We start from height 1 MAX tree lookup and then iteratively deepening it by two to always end at a MAX node
    current_height = 1

    # Set initial aggression level and max aggression level
    aggression_level = 1.0
    max_aggression_level = 1.0

    while True:
        # Expand the MAX tree for current player
        expand_result = expand_max_tree(result_store, board, N, player, timelimit, \
                                        0, current_height, -math.inf, math.inf, \
                                        aggression_level, max_aggression_level)

        # Yield the result
        yield expand_result[0]

        # Increase the tree height and expand again
        current_height += 1

        # Get the new time diff
        time_elapsed = time.time() - start_time

        # Calculate aggression level
        aggression_level = aggression_level * (1 + time_elapsed / timelimit)

        # Control aggression within limits :P
        if aggression_level > 1.0:
            aggression_level = 1.0


if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise Exception("Usage: Raichu.py N player board timelimit")
        
    (_, N, player, board, timelimit) = sys.argv
    N=int(N)
    timelimit=int(timelimit)
    if player not in "wb":
        raise Exception("Invalid player.")

    if len(board) != N*N or 0 in [c in "wb.WB@$" for c in board]:
        raise Exception("Bad board string.")

    print("Searching for best move for " + player + " from board state: \n" + board_to_string(board, N))
    print("Here's what I decided:")
    for new_board in find_best_move(board, N, player, timelimit):
        print(new_board)
