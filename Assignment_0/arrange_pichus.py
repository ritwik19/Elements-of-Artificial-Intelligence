#!/usr/local/bin/python3
#
# arrange_pichus.py : arrange agents on a grid, avoiding conflicts
#
# Submitted by : [RITWIK BUDHIRAJA rbudhira@iu.edu]
#
# Based on skeleton code in CSCI B551, Fall 2021.

import sys

# Parse the map from a given filename
def parse_map(filename):
	with open(filename, "r") as f:
		return [[char for char in line] for line in f.read().rstrip("\n").split("\n")][3:]

def cond_checker(house_map, r, c):
    r_dim = len(house_map)
    c_dim = len(house_map[0])
    check_row = True
    check_column = True
    check_diagonal = True

    if(house_map[r][c] != '.'):
        return False
    else:

        # Check along the row r
        in_row = [j for j in range(c_dim) if house_map[r][j] == 'p']
        if len(in_row) > 0:
            for k in in_row:
                min_el, max_el = min(k, c), max(k, c)
                has_walls = False
                for ci in range(min_el + 1,max_el):
                    has_walls = has_walls or house_map[r][ci] in 'X@'
                    if has_walls:
                        break
                check_row = check_row and has_walls

        # Check along the column c
        in_col = [i for i in range(r_dim) if house_map[i][c] == 'p']
        if len(in_col) > 0:
            for k in in_col:
                min_el, max_el = min(k, r), max(k, r)
                has_walls = False
                for ri in range(min_el + 1,max_el):
                    has_walls = has_walls or house_map[ri][c] in 'X@'
                    if has_walls:
                        break
                check_column = check_column and has_walls

        # The concept of the diagonal was referred to from http://dvatvani.github.io/8-Queens.html
        in_diagonal = [(ri, ci) for ri in range(0, r_dim) for ci in range(0, c_dim) if (house_map[ri][ci] == 'p' and abs(r - ri) == abs(c - ci))]

        # Function to check sign of number
        def sign(n):
            return 1 if n >= 0 else -1

        # Check along either diagonal with slope 1 or -1
        if len(in_diagonal) > 0:
            for pos in in_diagonal:
                diags = [x for x in zip(list(range(r, pos[0], sign(pos[0] - r))), list(range(c, pos[1], sign(pos[1] - c)))) if x not in [(r, c), pos]]
                has_walls = False
                for check_pos in diags:
                    has_walls = has_walls or (house_map[check_pos[0]][check_pos[1]] == 'X')
                    if has_walls:
                        break
                check_diagonal = check_diagonal and has_walls

        # Combine all flags
        return check_row and check_column and check_diagonal

# Count total # of pichus on house_map
def count_pichus(house_map):
    return sum([ row.count('p') for row in house_map ] )

# Return a string with the house_map rendered in a human-pichuly format
def printable_house_map(house_map):
    return "\n".join(["".join(row) for row in house_map])

# Add a pichu to the house_map at the given position, and return a new house_map (doesn't change original)
def add_pichu(house_map, row, col):
    return house_map[0:row] + [house_map[row][0:col] + ['p',] + house_map[row][col+1:]] + house_map[row+1:]

# Get list of successors of given house_map state
def successors(house_map):
    return [add_pichu(house_map, r, c) for r in range(0, len(house_map)) for c in range(0,len(house_map[0])) if cond_checker(house_map,r,c)]

# check if house_map is a goal state
def is_goal(house_map, k):
    return count_pichus(house_map) == k 

# Arrange agents on the map
#
# This function MUST take two parameters as input -- the house map and the value k --
# and return a tuple of the form (new_house_map, success), where:
# - new_house_map is a new version of the map with k agents,
# - success is True if a solution was found, and False otherwise.
#
def solve(initial_house_map,k):
    fringe = [initial_house_map]
    while len(fringe) > 0:
        current = fringe.pop()
        if is_goal(current, k):
            return(current, True)
        for new_house_map in successors(current):
            fringe.append(new_house_map)
    return False


# Main Function
if __name__ == "__main__":
    house_map=parse_map(sys.argv[1])
    # This is k, the number of agents
    k = int(sys.argv[2])
    print ("Starting from initial house map:\n" + printable_house_map(house_map) + "\n\nLooking for solution...\n")
    solution = solve(house_map,k)
    print ("Here's what we found:")
    if (solution == False):
        print(solution)
    else:
        print (printable_house_map(solution[0]))
