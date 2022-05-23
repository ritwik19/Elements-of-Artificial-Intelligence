#!/usr/local/bin/python3
# assign.py : Assign people to teams
#
# Code by: Harsh Srivastava <hsrivas>
#        : Ritwik Budhiraja <rbudhira>
#        : Yash Kalpesh Shah <yashah>
#
# Based on skeleton code by D. Crandall and B551 Staff, September 2021
#

import copy

# Import the 'heapq' module
#   The heapq module is Python's in-built implementation of a priority queue as per the following doc-
#       Refer to - https://docs.python.org/3/library/heapq.html
import heapq

# Import the 'itertools' module
#   The itertools module provides a lot of fast and efficient ways to handle permutations and combinations.
#       We will be using this module to obtain the combinations of students in pairs of 1, 2 and 3
#       Refer to - https://docs.python.org/3/library/itertools.html#itertools.combinations
import itertools

import math
import sys

def read_student_preferences(input_file):
    """
    @brief
    This method reads the student preferences from the input file provided.

    @desc
    The output is in the following format:
    {
        "username1": ("username1", "team-requested", "objection-list")
    }

    @returns Student Preference Sample space
    """
    student_preferences = {}
    lines = open(input_file, "r").readlines()
    for line in lines:
        line = line.strip()
        if len(line) == 0:
            continue

        splits = line.split()
        team_request = splits[1].split("-")
        objection_list = splits[2].split(",")
        team_request.remove(splits[0])
        student_preferences[splits[0]] = (splits[0], team_request, objection_list, [])
    return student_preferences

"""
Following are the helper methods to access values from tuple and also to improve readability.
"""
def get_studentname(student_state: list):
    return student_state[0]

def get_team_request(student_state: list):
    return student_state[1]

def get_team_request_count(student_state: list):
    return len(student_state[1])

def get_objection_list(student_state: list):
    return student_state[2]

def get_objection_list_count(student_state: list):
    return len(student_state[2])

def get_current_team(student_state: list):
    return student_state[3]

def get_current_team_count(student_state: list):
    return len(student_state[3])

def calculate_member_cost(student_state: list, team: list = None):
    """
    @brief
    Calculates the time required to tend to the mails sent by the user in case any conflits are observed.
    The time calculated is relative and doesn't take into account the one time assignment cost.

    @desc
    It will take care of the time needed to tackle the following-
        1. Team size mismatch
        2. Students not getting the students they wanted on their team
        3. Students getting a person/s they didn't want to work with
    """
    if team == None:
        team = get_current_team(student_state)

    cost = 0.0

    # Add the cost of team size mismatch
    cost += 2.0 if (len(get_team_request(student_state)) + 1) != len(team) else 0.0

    # Calculate and add the cost of a person not getting the people they requested
    #   against the probability that they still shared code (60*0.05=3)
    cost += len([x for x in get_team_request(student_state) if x not in team and x not in ["xxx", "zzz"]]) * 3

    # Adding cost for the student complaining to the Dean if they got matched with
    #   someone on the objection list
    cost += len([x for x in get_objection_list(student_state) if x in team]) * 10.0

    # Return the calculated cost
    return cost

def calculate_relative_total_cost(student_preferences: list, team: list):
    """
    @brief
    Calculates the total time that the formed team will add to the total effort if any.
    """
    # Add 5.0 which is one team cost
    total = 5.0

    # Check relative team member costs
    for member in team:
        total += calculate_member_cost(student_preferences[member], team=team)
    return total

def calculate_fringe_state_cost(student_preferences: list, fringe_state: list):
    """
    @brief
    Calculates the cost of a particular arrangement of teams also called the fringe_state.
    """
    return sum([calculate_relative_total_cost(student_preferences, team) for team in fringe_state])

def is_goal(students: list, fringe_state: list):
    """
    @brief
    Checks whether all students are in the fringe_state.

    @desc
    Once all students are in the fringe_state the algorithm will yield it as a possible result,
        if the fringe_state cost is lesser than previous costs yielded.

    @returns bool, bool -> If it contains all students in the arragement, Remainder student if any
    """
    students = copy.copy(students)
    for team in fringe_state:
        for student in team:
            students.remove(student)
    return len(students) == 0, students

class FringeState:
    """
    @brief
    Simple plain old Python object to help heapify sorting
    """
    def __init__(self, groups: list, cost: float) -> None:
        self.groups = groups
        self.cost = cost

    def __lt__(self, value):
        """
        This function is used to help Python's heapq.heapify function
        """
        return self.cost < value.cost

def solver(input_file):
    """
    1. This function should take the name of a .txt input file in the format indicated in the assignment.
    2. It should return a dictionary with the following keys:
        - "assigned-groups" : a list of groups assigned by the program, each consisting of studentnames separated by hyphens
        - "total-cost" : total cost (time spent by instructors in minutes) in the group assignment
    3. Do not add any extra parameters to the solver() function, or it will break our grading and testing code.
    4. Please do not use any global variables, as it may cause the testing code to fail.
    5. To handle the fact that some problems may take longer than others, and you don't know ahead of time how
       much time it will take to find the best solution, you can compute a series of solutions and then
       call "yield" to return that preliminary solution. Your program can continue yielding multiple times;
       our test program will take the last answer you 'yielded' once time expired.
    """

    # Read student preferences
    student_preferences = read_student_preferences(input_file)

    # Get the list of students
    students = [x for x in student_preferences]

    # Create a fringe with an empty fringe state and its cost
    fringe = [ FringeState([], calculate_fringe_state_cost(student_preferences, [])) ]

    # Dictionary to store the minimum goal state received so far
    minimum_goal = {
            "assigned-groups": [],
            "total-cost": math.inf
        }

    # Loop until the fringe is empty
    while len(fringe) > 0:
        # Pop the item from the heapq
        fringe_last: FringeState = heapq.heappop(fringe)

        # Check if is_goal() and store the remainder students list
        is_goal_bool, remainder_students = is_goal(students, fringe_last.groups)
        if is_goal_bool:
            current_goal = {
                "assigned-groups": ["-".join(x) for x in fringe_last.groups],
                "total-cost": calculate_fringe_state_cost(student_preferences, fringe_last.groups) # fringe_last["cost"]
            }

            # Yield the result if it is better than the last minimum
            if current_goal["total-cost"] < minimum_goal["total-cost"]:
                minimum_goal = current_goal
                yield minimum_goal

        # Generate combinations of 1, 2 and 3 student teams from the remainder_students
        #   Note: In the first pass the remainder students list will be all the students
        combinations = list(itertools.combinations(remainder_students, 1)) \
                     + list(itertools.combinations(remainder_students, 2)) \
                     + list(itertools.combinations(remainder_students, 3))

        # Map all combinations to FringeState objects
        combinations = list(map(lambda x: FringeState([x], calculate_relative_total_cost(student_preferences, x)), combinations))

        # Start from the biggest teams first to increase chances of lower cost
        #   More people in the teams, the lower time it takes to check the assignments
        combinations.reverse()

        # Define a threshold 't' for picking that many items having the lowest cost
        #   This is more of a hard cost threshold
        minimum_combinations = []
        if len(combinations) > 0:
            minimum_cost = combinations[0].cost
            threshold = (minimum_cost * 1.3)
            while len(combinations) > 0 and combinations[0].cost <= threshold:
                minimum_combinations.append(combinations.pop(0))

        # Transform the combinations and add to the fringe with extended teams from each combination
        fringe = fringe + list(map(lambda x: FringeState(fringe_last.groups + x.groups,
                                            calculate_fringe_state_cost(student_preferences, fringe_last.groups + x.groups)),
                                    minimum_combinations))

        # Calculate the heapify function to maintain heap invariant
        heapq.heapify(fringe)

if __name__ == "__main__":
    if(len(sys.argv) != 2):
        raise(Exception("Error: expected an input filename"))

    for result in solver(sys.argv[1]):
        print("----- Latest solution:\n" + "\n".join(result["assigned-groups"]))
        print("\nAssignment cost: %d \n" % result["total-cost"])
    
