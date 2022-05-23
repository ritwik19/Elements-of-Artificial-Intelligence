# Simple quintris program! v0.2
# D. Crandall, Sept 2021

from typing import Counter
from AnimatedQuintris import *
from SimpleQuintris import *
from kbinput import *
import time, sys
import numpy
import math

# Just flip DEBUG to True for enabling debug output
DEBUG = False

# Custom printd for debug
def printd(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

class HumanPlayer:
    def get_moves(self, quintris):
        print("Type a sequence of moves using: \n  b for move left \n  m for move right \n  n for rotation\n  h for horizontal flip\nThen press enter. E.g.: bbbnn\n")
        moves = input()
        return moves

    def control_game(self, quintris):
        while 1:
            c = get_char_keyboard()
            commands =  { "b": quintris.left, "h": quintris.hflip, "n": quintris.rotate, "m": quintris.right, " ": quintris.down }
            commands[c]()

#####
# This is the part you'll want to modify!
# Replace our super simple algorithm with something better
#
class ComputerPlayer:
    def __init__(self):
        self.current_moves = []

    def grab_matrix(self, board: list, from_point: tuple, to_point: tuple):
        # Helper method to grab a matrix from a bigger matrix
        if not (to_point[0] < len(board) and to_point[1] < len(board[0])):
            return None
        return [[(1 if board[i][j] == "x" else 0) \
                    for j in range(from_point[1], to_point[1] + 1)] \
                        for i in range(from_point[0], to_point[0] + 1)]

    def apply_mask(self, mask: list, target: list):
        # Apply a mask with the multiplication operator
        return [tuple([mask[i][j] * target[i][j] for j in range(len(mask[i]))])
                                                for i in range(len(mask))]

    # Code borrowed from the link below.
    #   (A really neat logic to rotate matrices 90-deg clockwise in Python)
    #   - https://stackoverflow.com/questions/8421337/rotating-a-two-dimensional-array-in-python
    def rotate_piece_mat(self, piece_mat: list):
        # Rotate the matrix and return result
        return list(zip(*piece_mat[::-1]))

    def flip_horizontal_piece_mat(self, piece_mat: list):
        # Flip a matrix horizontally
        return [row[::-1] for row in piece_mat] 

    def complement_binary_matrix(self, mat: list):
        # Complement a binary matrix
        return [[0 if mat[i][j] == 1 else 1 \
                    for j in range(0, len(mat[i]))] \
                        for i in range(0, len(mat))]

    def convert_piece_to_matrix(self, piece: list):
        # Convert a piece with "x" and " " to a binary matrix
        return [tuple([(1 if piece[i][j] == "x" else 0) \
                    for j in range(0, len(piece[i]))]) \
                        for i in range(0, len(piece))]

    def convert_matrix_to_piece(self, piece_mat: list):
        # Convert a binary matrix to a "x" and " " piece
        return ["".join([("x" if piece_mat[i][j] == 1 else " ") \
                    for j in range(0, len(piece_mat[i]))]) \
                        for i in range(0, len(piece_mat))]

    def does_piece_fit(self, piece: list, target: list):
        # Check if the piece fits the given target
        #   - Target should be same size as piece
        complement_target = self.complement_binary_matrix(target)
        mask_on_target = self.apply_mask(piece, complement_target)
        return piece == mask_on_target

    def count_blanks(self, piece_row: list, target_row: list):
        # Count the blanks in each row
        return sum([1 if (piece_row[i] == target_row[i] and piece_row[i] == 0) else 0 \
                    for i in range(len(piece_row))])

    def get_loss(self, piece_mat: list):
        # Get the loss that occurs through a matrix
        #   - Loss happens when a blank space is below another "x" tile in a piece
        loss = 0
        for r in range(len(piece_mat) - 1):
            for c in range(len(piece_mat[0])):
                if piece_mat[r][c] > piece_mat[r + 1][c] \
                    or (piece_mat[r][c] == 0 and piece_mat[r + 1][c] == 0):
                    loss += 1
        return loss

    def get_rows_with_gain(self, piece_mat: list):
        # Get the gain that occurs through a matrix
        #   - Gain happens when a blank space is above another "x" tile in a piece
        #     so the other tiles falling can possibly fit in there
        i = 0
        while i < len(piece_mat) and len("".join(piece_mat[i]).replace("x", "")) != 0:
            i += 1
        return range(i)

    def piece_mask_cost(self, quintris: QuintrisGame, current_piece, current_piece_matrix, r, c):
        # Heuristic: 1
        # Calculate the gain and loss by each piece, while also checking if the piece fits
        #   If the piece doesn't fit then return inf loss and gain
        height = len(current_piece_matrix)
        width = len(current_piece_matrix[0])
        target_area = self.grab_matrix(quintris.state[0], (r, c), (r + height - 1, c + width - 1))
        piece_fits = self.does_piece_fit(current_piece_matrix, target_area)
        if target_area == None or (target_area != None and not piece_fits):
            return (-math.inf, math.inf)
        else:
            rows_with_gain = self.get_rows_with_gain(current_piece)
            gain = sum([self.count_blanks(current_piece_matrix[r], target_area[r]) for r in rows_with_gain])

            loss = self.get_loss(current_piece)
            return (loss, gain)

    def piece_row_elimination_gain(self, current_state, current_piece, current_piece_matrix, r, c, new_board):
        # Heuristic: 2
        # Count the row elimination totals
        #   Basically count the number of "x" in each row the piece sits in
        #   This acts as a heuristic for ensuring that pieces that cut more rows are chosen
        if ("x" * len(new_board[0])) in new_board:
            return [quintris.BOARD_WIDTH * 3]

        piece_row_elimination_gain_list = []

        for piece_r in range(len(current_piece_matrix)):
            target_r = r + piece_r
            current_row_count_x = new_board[target_r].count("x")

            # Unused formulas for effective gain, we chose to use simply the count "x" in the row at the end
            # effective_gain = current_row_count_x * (current_row_count_x - current_state[0][target_r].count("x"))
            # effective_gain = current_row_count_x - current_state[0][target_r].count("x")

            piece_row_elimination_gain_list.append(current_row_count_x)

        # We were sorting this and returning the max at a point earlier, but we now sum it up outside when used
        # piece_row_elimination_gain_list.sort(key=lambda x: -x[1])
        return piece_row_elimination_gain_list

    def get_shadow_value(self, new_board: list, current_piece: list, target_r, target_c):
        # Heuristic: 3
        # Get the blank spaces underneath the pieces

        # Find the row with all "x" in piece
        for r in range(len(current_piece) - 1, -1, -1):
            if len(current_piece[r].replace("x", "")) == 0:
                break

        # Sum all blank tile count below that row in the new_board
        shadow_value = 0
        for shadow_r in range(target_r + r, len(new_board)):
            shadow_value += new_board[shadow_r][target_c:target_c + len(current_piece[0])].count(" ")

        return shadow_value

    def get_stack_height(self, new_board: list):
        # Heuristic: 4
        # Get the max height the tiles have reached so far
        #   This controls the amount of height a piece adds to the board, as that is usually disastrous
        rotated = self.rotate_piece_mat(new_board)
        return max([rotated[c].count("x") for c in range(len(rotated))])

    def cost_combinator(self, loss, gain, row_elimination, shadowed_blanks, stack_height):
        # Old formula to combine the heuristics
        # 120 is the max shadow value used to pad the other values
        # cost = shadowed_blanks
        # cost -= 120 * row_elimination
        # cost += 120 * stack_height * loss
        # cost -= 120 * stack_height * gain

        # New formula to combine heuristics, works relatively better
        cost = shadowed_blanks # Simply add the shadows
        cost -= row_elimination # Subtract the rows eliminated
        cost += stack_height * loss # Add the loss as the cost increases with loss
        cost -= stack_height * gain # Subtract the gain as the cost reducec with the gain

        # Return the values with the components
        return {
            "value": cost,
            "loss": loss,
            "gain": gain,
            "row_elimination": row_elimination,
            "shadowed_blanks": shadowed_blanks
        }

    def calculate_cost(self, quintris: QuintrisGame, piece_mask_cost_rc, piece_row_elimination_gains, shadowed_blanks, stack_height):
        # Calculate the total cost incurred by piece
        loss, gain = piece_mask_cost_rc
        row_elimination = sum(piece_row_elimination_gains)

        if row_elimination == quintris.BOARD_WIDTH:
            return self.cost_combinator(-math.inf, math.inf, math.inf, -math.inf, quintris.BOARD_HEIGHT)

        return self.cost_combinator(loss, gain, row_elimination, shadowed_blanks, stack_height)

    def find_possible_mask_applications(self, quintris: QuintrisGame, current_state: list, prospective_piece: list, silent=False):
        # This method goes through each row until the piece starts to fit in all columns, basically if it fits everywhere that is the time to stop computing

        current_board = current_state[0]

        # Find all possible transformations for the piece, we only flip once and rotate once for each original transformation
        possible_transformations = [(self.convert_piece_to_matrix(prospective_piece), "")]
        for i in range(0, 3):
            possible_transformations += [(self.rotate_piece_mat(possible_transformations[-1][0]), possible_transformations[-1][1] + "n")]
        for i in range(0, 3):
            possible_transformations += [(self.flip_horizontal_piece_mat(possible_transformations[i][0]), possible_transformations[i][1] + "h")]
            for i in range(0, 3):
                possible_transformations += [(self.rotate_piece_mat(possible_transformations[-1][0]), possible_transformations[-1][1] + "n")]

        # Remove duplicate states
        possible_transformations_set = set()
        possible_transformations_temp = []
        for possible_pair in possible_transformations:
            hashable = "\n".join(["".join(["x" if col == 1 else " " for col in row]) for row in possible_pair[0]])
            if hashable not in possible_transformations_set:
                possible_transformations_temp.append(possible_pair)
                possible_transformations_set.add(hashable)
        possible_transformations = possible_transformations_temp
        del possible_transformations_temp
        del possible_transformations_set

        # Store the moves in a list
        moves = []

        # Loop through each transformation
        for possible_pair in possible_transformations:
            rotation = possible_pair[0]

            # Conver to the piece "x"/" " form
            piece_rotation = self.convert_matrix_to_piece(rotation)
            for target_r in range(quintris.BOARD_HEIGHT - len(piece_rotation), -1, -1):
                all_fit = True

                for target_c in range(0, quintris.BOARD_WIDTH - len(piece_rotation[0]) + 1):
                    piece_mask_cost_rc = self.piece_mask_cost(quintris, piece_rotation, rotation, target_r, target_c)

                    if piece_mask_cost_rc == (-math.inf, math.inf):
                        all_fit = False
                    else:
                        new_state = quintris.place_piece(current_board, 0, piece_rotation, target_r, target_c)
                        stack_height = self.get_stack_height(new_state[0])
                        piece_row_elimination_gains = self.piece_row_elimination_gain(current_state, piece_rotation, rotation, target_r, target_c, new_state[0])
                
                        shadowed_blanks = self.get_shadow_value(new_state[0], piece_rotation, target_r, target_c)

                        diff = target_c - quintris.col
                        cost = self.calculate_cost(quintris, piece_mask_cost_rc, piece_row_elimination_gains, shadowed_blanks, stack_height)
                        if cost in [-math.inf, math.inf]:
                            continue
                        elif cost == 0:
                            cost = -math.inf
                        if diff < 0:
                            moves.append({
                                "move": possible_pair[1] + "b" * abs(diff),
                                "cost": cost,
                                "piece": piece_rotation,
                                "target_r": target_r,
                                "target_c": target_c,
                                "new_state": new_state
                            })
                        elif diff > 0:
                            moves.append({
                                "move": possible_pair[1] + "m" * abs(diff),
                                "cost": cost,
                                "piece": piece_rotation,
                                "target_r": target_r,
                                "target_c": target_c,
                                "new_state": new_state
                            })

                if all_fit:
                    break
        
        # Sort the moves by cost
        moves.sort(key=lambda x: x["cost"]["value"])

        return moves

    # This function should generate a series of commands to move the piece into the "optimal"
    # position. The commands are a string of letters, where b and m represent left and right, respectively,
    # and n rotates. quintris is an object that lets you inspect the board, e.g.:
    #   - quintris.col, quintris.row have the current column and row of the upper-left corner of the 
    #     falling piece
    #   - quintris.get_piece() is the current piece, quintris.get_next_piece() is the next piece after that
    #   - quintris.left(), quintris.right(), quintris.down(), and quintris.rotate() can be called to actually
    #     issue game commands
    #   - quintris.get_board() returns the current state of the board, as a list of strings.
    #
    def get_moves(self, quintris: QuintrisGame, is_animated=False):
        # get all immediate moves
        all_current_applications = self.find_possible_mask_applications(quintris, quintris.state, quintris.piece, silent=True)

        # If the board is animated then consider the future moves based on next_piece
        if is_animated:
            for current_application in all_current_applications:
                all_future_applications = self.find_possible_mask_applications(quintris, current_application["new_state"], quintris.next_piece, silent=True)
                current_application["cost"]["value"] += min([x["cost"]["value"] for x in all_future_applications])
            all_current_applications.sort(key=lambda x: x["cost"]["value"])
        return all_current_applications[0]["move"]
       
    # This is the version that's used by the animted version. This is really similar to get_moves,
    # except that it runs as a separate thread and you should access various methods and data in
    # the "quintris" object to control the movement. In particular:
    #   - quintris.col, quintris.row have the current column and row of the upper-left corner of the 
    #     falling piece
    #   - quintris.get_piece() is the current piece, quintris.get_next_piece() is the next piece after that
    #   - quintris.left(), quintris.right(), quintris.down(), and quintris.rotate() can be called to actually
    #     issue game commands
    #   - quintris.get_board() returns the current state of the board, as a list of strings.
    #
    def control_game(self, quintris: QuintrisGame):
        # Get the move from the method for simple game and move it manually
        while 1:
            time.sleep(0.1)

            if len(self.current_moves) == 0:
                self.current_moves = self.get_moves(quintris, is_animated=True)
            else:
                move = self.current_moves[0]
                self.current_moves = self.current_moves[1:]
                if move == "h":
                    quintris.hflip()
                elif move == "n":
                    quintris.rotate()
                elif move == "b":
                    quintris.left()
                elif move == "m":
                    quintris.right()
                if len(self.current_moves) == 0:
                    quintris.down()


###################
#### main program

(player_opt, interface_opt) = sys.argv[1:3]

try:
    if player_opt == "human":
        player = HumanPlayer()
    elif player_opt == "computer":
        player = ComputerPlayer()
    else:
        print("unknown player!")

    if interface_opt == "simple":
        quintris = SimpleQuintris()
    elif interface_opt == "animated":
        quintris = AnimatedQuintris()
    else:
        print("unknown interface!")

    quintris.start_game(player)

except EndOfGame as s:
    print("\n\n\n", s)



