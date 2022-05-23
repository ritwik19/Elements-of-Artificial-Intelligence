#!/usr/local/bin/python3
# solver2021.py : 2021 Sliding tile puzzle solver
#
# Code by: Harsh Srivastava <hsrivas>
#        : Ritwik Budhiraja <rbudhira>
#        : Yash Kalpesh Shah <yashah>
#
# Based on skeleton code by D. Crandall & B551 Staff, September 2021
#

import sys
from datetime import datetime

# Import the 'heapq' module
#   The heapq module is Python's in-built implementation of a priority queue as per the following doc-
#       Refer to - https://docs.python.org/3/library/heapq.html
import heapq

ROWS=5
COLS=5

def printable_board(board):
    return [ ('%3d ')*COLS  % board[j:(j+COLS)] for j in range(0, ROWS*COLS, COLS) ]

def highlight_printable(board, fn):
    string = ""
    for i in range(0, 5):
        for j in range(0, 5):
            if fn(i, j):
                string += "\033[45;37;1m" + ("%3d " % board[i * 5 + j]) + "\033[0m"
            else:
                string += ("%3d " % board[i * 5 + j])
        string += "\n"
    return string
    
"""
Implementing a python class representation for the Puzzle 2021 board State at any given point throughout the search. 
Here's what this class does:
    - The Puzzle2021State stores the current board along with the previous number of moves performed to reach that state. 
    - An important part to note is that we pass 'g' and 'h' heuristic functions as parameters to the constructor class 
        at any point to change the heuristics dynamically.
    - This class also has the `move_board` function which performs a particular move on the board.
    - 'g' and 'h' are heuristic values which are pre-cached to improve the overall time complexity by making faster computations. 
    - The class Puzzle2021State also has multi-helper transform function to transform the currennt board into a new board 
        which is used in the 'successors' function later in the code to generate all the possible successor states from any given state.
"""
class Puzzle2021State:
    """
    This function takes in the board, a list of traversed moves, previous distance variable, g_heuristic, and h_heuristic as parameters. 
    It then intializes these values to the data members of the class.
    """
    def __init__(self, board, moves = [], prev_dist = 0, g_heuristic_wrapper = None, h_heuristic_wrapper = None):
        self.board = board
        self.moves = moves
        self.total_distance_from_initial = prev_dist + 1

        self.g_heuristic_wrapper = g_heuristic_wrapper
        if self.g_heuristic_wrapper == None:
            self.g_heuristic_wrapper = lambda state: 0

        self.h_heuristic_wrapper = h_heuristic_wrapper
        if self.h_heuristic_wrapper == None:
            self.h_heuristic_wrapper = lambda state: 0

        self.g = self.g_heuristic_wrapper(self)
        self.h = self.h_heuristic_wrapper(self)
        self.f = self.g + self.h

    def get_board(self):
        """
        This is a getter function used to get the current configuration of the board.
        """
        return self.board

    def get_moves(self):
        """
        This is a getter function used to get the current list of traversed moves.
        """
        return self.moves

    def __lt__(self, value):
        """
        This function overrides Python's inbuilt less than (`<`) symbol in order to compare any two states. 
        This proves to be really helpful when using the 'heapify' function from the 'heapq' library of python. This 'heapq'
        library provides access to a faster implementation of the priority queue. 
        """
        return self.f < value.f

    def move_board(self, move):
        """
        This function gets the next move to be performed as a parameter and then changes
            the configuration of the board according to the move. 
        It also then calculates the g_heuristic and the h_heuristic at this point.
        """
        new_board = None
        if move[0] == "L":
            new_board = self.transform_shift_row(int(move[1:]) - 1, -1)
        elif move[0] == "R":
            new_board = self.transform_shift_row(int(move[1:]) - 1, 1)
        elif move[0] == "U":
            new_board = self.transform_shift_column(int(move[1:]) - 1, -1)
        elif move[0] == "D":
            new_board = self.transform_shift_column(int(move[1:]) - 1, 1)
        elif move == "Oc":
            new_board = self.transform_rotate_outer_c()
        elif move == "Occ":
            new_board = self.transform_rotate_outer_cc()
        elif move == "Ic":
            new_board = self.transform_rotate_inner_c()
        elif move == "Icc":
            new_board = self.transform_rotate_inner_cc()

        return Puzzle2021State(new_board,
                                self.moves + [move],
                                prev_dist=self.total_distance_from_initial,
                                g_heuristic_wrapper=self.g_heuristic_wrapper,
                                h_heuristic_wrapper=self.h_heuristic_wrapper)

    """
    Transformation methods for the board. 
    These methods correspond to each transformation required in the board. They are as follows:
        - For sliding rows, R (right) or L (left), followed by the row number indicating the row to move left or
          right. The row numbers range from 1-5.
        - For sliding columns, U (up) or D (down), followed by the column number indicating the column to move
          up or down. The column numbers range from 1-5.
        - For rotations, I (inner) or O (outer), followed by whether the rotation is clockwise (c) or counterclock-
          wise (cc).
    The `solve()` function returns a list of valid moves which are encoded as strings corresponding to the 
    methods shown above.
    """
    def transform_rotate_outer_c(self):
        new_board = [x for x in self.board]

        # The transformation mappings are hardcoded for the outer clockwise rotation
        for (orig, dest) in zip([0, 1, 2, 3, 4, 9, 14, 19, 24, 23, 22, 21, 20, 15, 10, 5],
                                [1, 2, 3, 4, 9, 14, 19, 24, 23, 22, 21, 20, 15, 10, 5, 0]):
            new_board[dest] = self.board[orig]
        return new_board

    def transform_rotate_outer_cc(self):
        new_board = [x for x in self.board]

        # The transformation mappings are hardcoded for the outer counter-clockwise rotation
        for (orig, dest) in zip([1, 2, 3, 4, 9, 14, 19, 24, 23, 22, 21, 20, 15, 10, 5, 0],
                                [0, 1, 2, 3, 4, 9, 14, 19, 24, 23, 22, 21, 20, 15, 10, 5]):
            new_board[dest] = self.board[orig]
        return new_board

    def transform_rotate_inner_c(self):
        new_board = [x for x in self.board]

        # The transformation mappings are hardcoded for the inner clockwise rotation
        for (orig, dest) in zip([6, 7, 8, 13, 18, 17, 16, 11],
                                [7, 8, 13, 18, 17, 16, 11, 6]):
            new_board[dest] = self.board[orig]
        return new_board

    def transform_rotate_inner_cc(self):
        new_board = [x for x in self.board]

        # The transformation mappings are hardcoded for the inner counter-clockwise rotation
        for (orig, dest) in zip([7, 8, 13, 18, 17, 16, 11, 6],
                                [6, 7, 8, 13, 18, 17, 16, 11]):
            new_board[dest] = self.board[orig]
        return new_board

    def transform_shift_row(self, row, direction):
        new_board = [x for x in self.board]

        # Perform transformation by shifting each element on the row
        for c in range(COLS):
            new_board[row * COLS + c] = self.board[(row * COLS + ((c - direction) % COLS)) % (ROWS * COLS)]
        return new_board

    def transform_shift_column(self, column, direction):
        new_board = [x for x in self.board]

        # Perform transformation by shifting each element on the column
        for r in range(ROWS):
            new_board[r * COLS + column] = self.board[(((r - direction) % ROWS) * COLS + column) % (ROWS * COLS)]
        return new_board

# return a list of possible successor states
def successors(state):
    """
    This function returns a list of all possible successor states.

    All successor states are all the moves possible from the current state.
    """
    moves = ["L" + str(x) for x in list(range(1, 6))] \
          + ["R" + str(x) for x in list(range(1, 6))] \
          + ["U" + str(x) for x in list(range(1, 6))] \
          + ["D" + str(x) for x in list(range(1, 6))] \
          + ["Oc", "Occ", "Ic", "Icc"]
    states = [state.move_board(move) for move in moves]
    return states

# check if we've reached the goal
def is_goal(state):
    """
    This function checks if the current configuration of the board is the goal state or not.
    """
    return sum([state[i] == (i + 1) for i in range(0, ROWS * COLS)]) == (ROWS * COLS)

def board_to_np_array(board):
    """
    This function converts the data type of board to an array for an easier management and manipulation.
    """
    return [board[x * 5 : x * 5 + 5] for x in range(0, 5)]

def display_board(board, cursor, fn = lambda x: False):
    for i in range(5):
        print("\t", end="")
        for j in range(5):
            if cursor == [i, j]:
                print("\033[46;37;1m%3d \033[0m" % board[i * COLS + j], end="")
            elif fn(board[i * COLS + j]):
                print("\033[44;37;1m%3d \033[0m" % board[i * COLS + j], end="")
            else:
                print("%3d " % board[i * COLS + j], end="")
        print()

def misplaced_tiles(board, goal_state):
    """
    This function is used to calculate the number of misplaced tiles at each state 
    and then selects that specific state which has the least number of misplaced tiles.
    """
    count = 0
    for i in range(5):
        for j in range(5):
            count += board[i * COLS + j] != goal_state[i * COLS + j]
    return count

def permutation_inversions(board, return_board=False):
    """
    This function counts the number of tiles appearing after the current tile that has values lesser than the value at the current tile. 
    This operation gets such a count for each tile and then sums all the counts to get the final value. This heuristic then selects that specific state that has
    the least possible value of the sum.
    """
    new_board = [0 for x in board]
    for i in range(ROWS * COLS - 1):
        for j in range(i + 1, ROWS * COLS - 1):
            if board[i] > board[j]:
                new_board[i] += 1
    return new_board if return_board else sum(new_board)

# This method is not used in the final submission
def tangential_manhattan_wrap(board, goal_state):
    """
    Calculates the wrap around manhattan distances for each individual row, column.

    This was created as a solution to optimize the solution.
    """
    def wrap_distance(x1, x2, max_x):
        diff = x1 - x2
        wrap = diff + max_x if x1 < x2 else max_x - diff
        return wrap % max_x

    def calculate_horizontal_manhattan_wrap(row):
        target_goal_row = goal_state[row * COLS:row * COLS + COLS]
        board_row = board[row * COLS:row * COLS + COLS]
        wraps = [0 for x in target_goal_row]
        for i in range(len(board_row)):
            x = board_row[i]
            if x in target_goal_row:
                expected_i = target_goal_row.index(x)
                wraps[i] = wrap_distance(expected_i, i, COLS)
        return sum(wraps) / len(wraps)

    def calculate_vertical_manhattan_wrap(column):
        target_goal_column = goal_state[column:ROWS * (COLS - 1) + column:COLS]
        board_column = board[column:ROWS * (COLS - 1) + column:COLS]
        wraps = [0 for x in target_goal_column]
        for i in range(len(board_column)):
            x = board_column[i]
            if x in target_goal_column:
                expected_i = target_goal_column.index(x)
                wraps[i] = wrap_distance(expected_i, i, ROWS)
        return sum(wraps) / len(wraps)

    rows_manhattan_wrap = [calculate_horizontal_manhattan_wrap(i) for i in range(ROWS)]
    columns_manhattan_wrap = [calculate_vertical_manhattan_wrap(i) for i in range(COLS)]

    value = sum(rows_manhattan_wrap + columns_manhattan_wrap) / (len(rows_manhattan_wrap) + len(columns_manhattan_wrap))
    return value

def solve(initial_board):
    """
    1. This function should return the solution as instructed in assignment, consisting of a list of moves like ["R2","D2","U1"].
    2. Do not add any extra parameters to the solve() function, or it will break our grading and testing code.
       For testing we will call this function with single argument(initial_board) and it should return 
       the solution.
    3. Please do not use any global variables, as it may cause the testing code to fail.
    4. You can assume that all test cases will be solvable.
    5. The current code just returns a dummy solution.
    """

    # Create a goal state equivalent
    goal_state = list(range(1, ROWS * COLS + 1))

    # The cost function
    def g_function(state: Puzzle2021State):
        return state.total_distance_from_initial

    # The heuristic function
    def h_function(state: Puzzle2021State):
        return misplaced_tiles(state.get_board(), goal_state)

    # Check if already in goal state
    if is_goal(initial_board):
        return []

    # Wrap the current board in a Puzzle2021State
    initial_state = Puzzle2021State(initial_board,
                                    g_heuristic_wrapper=g_function,
                                    h_heuristic_wrapper=h_function)

    # Add the current state to the fringe
    fringe = [ initial_state ]

    # Using Algorithm #2 to find the optimal solution using A-star search
    while len(fringe) > 0:
        # Pop the lowest cost item form the fringe
        current = heapq.heappop(fringe)

        # Check if goal state is reached
        if is_goal(current.get_board()):
            # Return the moves
            return current.moves

        # Obtain all the successors
        for successor in successors(current):
            # Push them on the heap
            heapq.heappush(fringe, successor)
    return []

# Please don't modify anything below this line
#
if __name__ == "__main__":
    if(len(sys.argv) != 2):
        raise(Exception("Error: expected a board filename"))

    start_state = []
    with open(sys.argv[1], 'r') as file:
        for line in file:
            start_state += [ int(i) for i in line.split() ]

    if len(start_state) != ROWS*COLS:
        raise(Exception("Error: couldn't parse start state file"))

    print("Start state: \n" +"\n".join(printable_board(tuple(start_state))))

    print("Solving...")
    route = solve(tuple(start_state))
    
    print("Solution found in " + str(len(route)) + " moves:" + "\n" + " ".join(route))
