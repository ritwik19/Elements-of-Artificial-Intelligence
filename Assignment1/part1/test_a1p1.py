# !/usr/bin/env python3
# YOU SHOULD NOT MODIFY THIS FILE
# 
# test_a1p1.py version 2021.09.19
#
#Stephen Karukas, Zoher Kachwala, Vibhas Vats

#importing student's scripts
import solver2021
from collections import defaultdict
import pytest
import copy
import pdb
import numpy as np
from pprint import pprint
############ Tests ############
cstate = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24, 25]]
board=copy.deepcopy(cstate)
ROWS = 5
COLS = 5

def move_right(board, row):
  """Move the given row to one position right"""
  board[row] = board[row][-1:] + board[row][:-1]
  return board

def move_left(board, row):
  """Move the given row to one position left"""
  board[row] = board[row][1:] + board[row][:1]
  return board

def rotate_right(board,row,residual):
    board[row] = [board[row][0]] +[residual] + board[row][1:]
    residual=board[row].pop()
    return residual

def rotate_left(board,row,residual):
    board[row] = board[row][:-1] + [residual] + [board[row][-1]]
    residual=board[row].pop(0)
    return residual

def move_clockwise(board):
    """Move the outer ring clockwise"""
    board[0]=[board[1][0]]+board[0]
    residual=board[0].pop()
    board=transpose_board(board)
    residual=rotate_right(board,-1,residual)
    board=transpose_board(board)
    residual=rotate_left(board,-1,residual)
    board=transpose_board(board)
    residual=rotate_left(board,0,residual)
    board=transpose_board(board)
    return board

def move_cclockwise(board):
    """Move the outer ring counter-clockwise"""
    board[0]=board[0]+[board[1][-1]]
    residual=board[0].pop(0)
    board=transpose_board(board)
    residual=rotate_right(board,0,residual)
    board=transpose_board(board)
    residual=rotate_right(board,-1,residual)
    board=transpose_board(board)
    residual=rotate_left(board,-1,residual)
    board=transpose_board(board)
    return board

def transpose_board(board):
  """Transpose the board --> change row to column"""
  return [list(col) for col in zip(*board)]

def path_finder(board, path_):
  board = np.array(board).reshape(ROWS, COLS).tolist()
  """Uses the output of student's path to reach canonical state"""
  for direction in path_:
    if set(direction).intersection(set(['U','D','R','L'])):
        direction,index=direction
        index=int(index)-1
    if direction == "R":
        board = move_right(board, index)
    elif direction == "L":
        board = move_left(board, index)
    elif direction == "U":
        board = transpose_board(move_left(transpose_board(board), index))
    elif direction == "D":
        board = transpose_board(move_right(transpose_board(board), index))
    elif direction == 'Oc':
        board = move_clockwise(board)
    elif direction == 'Occ':
        board = move_cclockwise(board)
    elif direction == 'Ic':
        board=np.array(board)
        inner_board=board[1:-1,1:-1].tolist()
        inner_board = move_clockwise(inner_board)
        board[1:-1,1:-1]=np.array(inner_board)
        board=board.tolist()
    elif direction == 'Icc':
        board=np.array(board)
        inner_board=board[1:-1,1:-1].tolist()
        inner_board = move_cclockwise(inner_board)
        board[1:-1,1:-1]=np.array(inner_board)
        board=board.tolist()
  return board

def get_map_as_list(map_):
  ##Converts list of list into a single list for map
    map_out = []
    for row in map_:
        map_out.extend(row)
    return map_out

## Check the output of the puzzle
def check_puzzle(mapX,path_length):
    path_found=solver2021.solve(mapX)
    print(list(mapX))
    assert len(path_found)!=0, "No moves!"
    #valid path should be subset
    valid_moves_set={'R1','R2','R3','R4','R5','L1','L2','L3','L4','L5','U1','U2','U3','U4','U5','D1','D2','D3','D4','D5','Oc','Ic','Occ','Icc'}
    assert set(path_found).issubset(valid_moves_set), f"Invalid moves: {list(set(path_found) - valid_moves_set)}"
    # Solution found should be equal in length or less
    assert path_length >= len(path_found), "Suboptimal number of moves!"
    # Retrace to canonical form using student's path.
    assert cstate == path_finder(mapX, path_found), "Canonical state not reached. The moves do not solve the puzze!"

def load_map(fname):
    with open(fname,"r") as file:
        lines=file.read().splitlines()
        path_length=int(lines[0])
        mapX = []
        for line in lines[1:]:
            mapX += [ int(i) for i in line.split() ]
        return tuple(mapX), path_length

@pytest.mark.timeout(300)
def test_puzzle_2021_case1():
    mapX,path_length = load_map('test_board0.txt')
    check_puzzle(mapX,path_length)

@pytest.mark.timeout(900)
def test_puzzle_2021_case2():
    mapX,path_length = load_map('test_board0.5.txt')
    check_puzzle(mapX,path_length)