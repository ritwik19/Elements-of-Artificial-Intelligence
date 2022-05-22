#!/usr/local/bin/python3
#
# route_pichu.py : a maze solver
#
# Submitted by : [RITWIK BUDHIRAJA rbudhira@iu.edu]
#
# Based on skeleton code provided in CSCI B551, Fall 2021.

import sys

# Parse the map from a given filename
def parse_map(filename):
        with open(filename, "r") as f:
                return [[char for char in line] for line in f.read().rstrip("\n").split("\n")][3:]
                
# Check if a row,col index pair is on the map
def valid_index(pos, n, m):
        return 0 <= pos[0] < n  and 0 <= pos[1] < m

# Find the possible moves from position (row, col)
def moves(map, row, col):
        moves=((row+1,col), (row-1,col), (row,col-1), (row,col+1))

        # Return only moves that are within the house_map and legal (i.e. go through open space ".")
        return [ move for move in moves if valid_index(move, len(map), len(map[0])) and (map[move[0]][move[1]] in ".@" ) ]

# Perform search on the map
#
# This function MUST take a single parameter as input -- the house map --
# and return a tuple of the form (move_count, move_string), where:
# - move_count is the number of moves required to navigate from start to finish, or -1
#    if no such route exists
# - move_string is a string indicating the path, consisting of U, L, R, and D characters
#    (for up, left, right, and down)

def dir_route(curr_move, move, direc1):
        
       
        row_comp = curr_move[0] - move[0]
        col_comp = curr_move[1] - move[1]

        if (row_comp == 0 and col_comp == 1):
                direc1 = direc1 + 'L'
        elif (row_comp == 0 and col_comp == -1):
                direc1 = direc1 + 'R'
        elif (row_comp == 1 and col_comp == 0):
                direc1 = direc1 + 'U'
        elif (row_comp == -1 and col_comp == 0):
                direc1 = direc1 + 'D'
        

        return direc1


def search(house_map):
        # Find pichu start position
        pichu_loc=[(row_i,col_i) for col_i in range(len(house_map[0])) for row_i in range(len(house_map)) if house_map[row_i][col_i]=="p"][0]
        # person_loc=[(row_i,col_i) for col_i in range(len(house_map[0])) for row_i in range(len(house_map)) if house_map[row_i][col_i]=="@"][0]
        # print(pichu_loc)
        direc1 = ''
        fringe=[(pichu_loc,0, direc1)]
        #(row1, col1) = pichu_loc
        #(row2, col2) = person_loc
        vis_moves = []

        while fringe:
                # prev_move = curr_move
                (curr_move, curr_dist, direc1)=fringe.pop()
                #move_str=curr_path
                vis_moves.append(curr_move)
                # if (house_map[curr_mov[0]])
                # print(curr_move, curr_dist)
                valid_moves = moves(house_map, *curr_move)
                #if (not valid_moves):
                        #print(-1, ' Path not found')
                for move in valid_moves:
                        # print(move)
                        # if move  == prev_move:
                        #if (len(valid_moves) == 1):
                                #house_map[curr_move[0]][curr_move[1]] = 'X'

                        
                        
                        # elif (len(valid_moves)>1):
                                #if(all(elem in vis_moves for elem in valid_moves)):
                                        #pass
                                #elif (move in vis_moves):
                                        #continue
                        # if ((curr_move == pichu_loc and curr_dist > 0) or ((house_map[row1+1][col1] == 'X' or row1+1 == -1 or row1+1>len(house_map) or col1 == -1 or col1>len(house_map[0])) and (house_map[row1-1][col1] == 'X' or row1-1 == -1 or row1-1>len(house_map) or col1 == -1 or col1>len(house_map[0])) and (house_map[row1][col1+1] == 'X' or row1 == -1 or row1>len(house_map) or col1+1 == -1 or col1+1>len(house_map[0])) and (house_map[row1][col1-1] == 'X' or row1 == -1 or row1>len(house_map) or col1-1 == -1 or col1-1>len(house_map[0]))) or ((house_map[row2+1][col2] == 'X' or row2+1 == -1 or row2+1>len(house_map) or col2 == -1 or col2>len(house_map[0])) and (house_map[row2-1][col2] == 'X' or row2-1 == -1 or row2-1>len(house_map) or col2 == -1 or col2>len(house_map[0])) and (house_map[row2][col2+1] == 'X' or row2 == -1 or row2>len(house_map) or col2+1 == -1 or col2+1>len(house_map[0])) and (house_map[row2][col2-1] == 'X' or row2 == -1 or row2>len(house_map) or col2-1 == -1 or col2-1>len(house_map[0])))):
                                # return -1
                                
                        if house_map[move[0]][move[1]]=="@":

                                return ((curr_dist + 1, dir_route(curr_move, move, direc1)))
                                  # return a dummy answer
                        elif (move not in vis_moves):
                                fringe.append((move, curr_dist + 1, dir_route(curr_move, move, direc1)))


# Main Function
if __name__ == "__main__":
        house_map=parse_map(sys.argv[1])
        # print(house_map)
        print("Shhhh... quiet while I navigate!")
        solution = search(house_map)
        print("Here's the solution I found:")
        if (solution == None):
                print(-1, ' Path not found')
        else:
                print(str(solution[0]) + " " + solution[1])

