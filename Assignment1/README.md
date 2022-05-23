# A1: SEARCHING

## Authors
Harsh Srivastava <hsrivas@iu.edu> \
Ritwik Budhiraja <rbudhira@iu.edu> \
Yash Shah <yashah@iu.edu>

## Part 1: THE 2021 PUZZLE
1.  The search abstraction used in the Part 1 of the assignment is defined by the following information:
    -  **Set of Valid States**\
        The set of valid states for the given sample is the set of moves, which if taken will take us to the configuration of the goal state, given one move at a time. In other words, every possible legal move either shifting any row to the right or to the left, OR shifting any column up or down, OR shifting the elements in the outer ring clockwise or counter-clockwise, OR shifting the elements in the inner ring clockwise or counter-clockwise.  
        
    -   **Successor Function**\
        The successor function is the function `successors()` which takes in the current state of the board and then returns all the possible next moves as explained in the above paragraph.

    -   **Cost function**\
        The cost function is the sum of g_heuristic and h_heuristic functions. 
        -   `g_heuristic` gives us the cost which in this case 'number of moves' that we successively make until reaching the goal state which is incremented by `1` everytime we make a move. 
        -   `h_heuristic` gives us the Manhattan Distance of the current state to the goal state.

    -   **Goal State**\
        The goal state is the configuration of the board in which all the 25 values are arranged in an ascending order.

    -   **Initial State**\
        The initial state is the configuration of the board in which all the numbers are arranged randomly.

### Highlights for Part 1:
1. We implemented a python class representation for the `Puzzle2021State` at any given point throughout the search. The search algorithm used for the search abstraction for this problem is A*. The `Puzzle2021State` stores the current board along with the previous number of moves performed to reach that state. 

2. An important part to note is that we pass 'g_heuristic' and 'h_heuristic' functions as parameters to the constructor class at any point to change 
the heuristics dynamically.

3. 'g' and 'h' are heuristic values which are pre-cached to improve the overall time complexity by making faster computations. 

4. The class Puzzle2021State also has multi-helper transform functions to transform the current board into a new board which is used 
in the 'successors' function later in the code to generate all the possible successor states from any given state.

5. We have made use of the 'heapq' library which provides access to a faster implementation of the priority queue.

6. We have defined the `__lt__` function that overrides Python's inbuilt less than (`<`) symbol in order to compare any two states. This proves  to be really helpful when using the 'heapify' function from the 'heapq' library of python.

7. From the three possible heuristics of Manhattan Distance, No. of Misplaced Tiles, and Permutation Inversion, we chose and implemented the No. of Misplaced Tiles heuristic to solve the problem as it gave us the fastest computations on comparing the runtimes of each heuristic.

    As a part of our search for the heuristic, we also implemented a wrapping manhattan distance function, which calculates and averages the manhattan distance for every element in each row and column. However, it didn't turn out to be a good heuristic. Please refer to the `tangential_manhattan_wrap()` function.

### Questions:
1. In this problem, what is the branching factor of the search tree?
--> Since we can move 'Right' in 5 different rows, 'Left' in 5 different rows, 'Up' in 5 different columns, 'Down' in 5 different columns, 'Clockwise' in the inner loop and the outer loop and 'Counter-clockwise' in the inner loop and the outer loop, the branching factor is 24.

2. If the solution can be reached in 7 moves, about how many states would we need to explore before we found it if we used BFS instead of A* search? A rough answer is fine. 
--> The Breadth-First Search algorithm takes into account every successor state of the current given state. As mentioned above, we know that the branching factor for this search tree will be 24. So, if we are reaching the solution in 7 moves, we would have explored (24^8 - 1)/23 states if we had used BFS instead of the A* Search.

## Part 2: ROAD TRIP!
1.  The search abstraction used in the Part 2 of the assignment is defined by the following information:
    -  **Set of Valid States**\
        The set of valid states for the given problem is the set of cities and junctions, which if passed through will take us to the final destination.
        
    -   **Successor Function**\
        The successor function `successors()` takes in the current state(city/junction) on the map and then returns all the possible valid states(neighboring cities/junctions) from the current city/junction.

    -   **Cost function**\
        The cost function is the sum of g_heuristic and h_heuristic functions.

        We have 4 different pairs of (g+h) functions for each case-
        1. Segments
            - `g_heuristic` For segments the cost function is the cost for each move(i.e city to junction or vice-versa), which is constant here, that is `= 1`
            - `h_heuristic` We use the maximum_segment_length calculated while reading the segment data information, to calculate the possible segments to the goal state for the heuristic.

                Statistically, this will never overestimate the segments as the maximum_segment_count is used and it minimizes the heuristic.

                _Note: The `calculate_distance_haversine()` function is used to calculate the geographical/geodesic distance between two co-ordinates on the earth's surface._

        2. Distance
            - `g_heuristic` For distance the cost function is the distance for each segment, which is constant here, that is `= state.segment_length`
            - `h_heuristic` We use the `calculate_distance_haversine()` again to calculate the spherical distance between two locations.

                Since it is an approximately closest on-the-surface distance between two points, it will never overestimate the heuristic.

        3. Time
            - `g_heuristic` For time the cost function is the time taken if the person is traveling at the maximum speed limit for the segment.
            - `h_heuristic` We use the `calculate_distance_haversine()` again to calculate the spherical distance between two locations and then divide it with the maximum speed (`maximum_speed (of the whole dataset)` in case of initial state, else the maximum speed limit encountered during the whole route for the state being observed)

                Statistically, this will never overestimate the total time taken to reach the last node as the maximum speed limit is always used.

        3. Delivery
            - `g_heuristic` For time the cost function is the delivery time taken as calculated by the provided formula.

            - `h_heuristic` We use the `calculate_distance_haversine()` again to calculate the spherical distance between two locations.

                Once we have the distance, we get the value of the `t_trip` by dividing it with the overall maximum speed limit (if starting state) or the maximum speed limit for the route so far.

                Then we take the current segment's time taken to calculate the delivery time as if it were calculated the other way around, i.e. from the goal state to the staring point of the current segment.

                This gives us a great estimate to figure out the variable delivery times. However, the heuristic is not admissible as it might overestimate the delivery time at some points in time.

    -   **Goal State**\
        The goal state is the input given by the user which is going to be the ending city of the route.

    -   **Initial State**\
        The initial state is the input given by the user which is going to be the starting city of the route.

### Highlights for Part 2:
1. Like `Part 1`, we implemented a python class representation for the `RouteState` at any given point throughout the search. The search algorithm used for the search abstraction for this problem is A*. The `RouteState` stores the previous route taken along with the current city/junction the search is at.

    It also stores many other data points and keep calculating the various distance, speed and time related measures at init object time.

2. Similar to the `Part 1` the `g_heuristic` and `h_heuristic` functions are dynamic so the same class can be re-used for all four heuristic options.

3. We have defined the `__lt__` function that overrides Python's inbuilt less than (`<`) symbol in order to compare any two states. This proves  to be really helpful when using the 'heapify' function from the 'heapq' library of python.

4. Also overridden the `__hash__` and `__eq__` functions for similar sorting functions.


## Part 3: CHOOSING TEAMS
1.  The search abstraction used in the Part 3 of the assignment is defined by the following information:
    -  **Set of Valid States**\
        The set of valid states for the given sample is any arrangement of the students to form a team of one, two or three.  
        
    -   **Successor Function**\
        The successors are obtained by appending individual team combinations obtained using the Python's `itertools.combinations()` method

    -   **Cost function**\
        The cost function is the sum of cost of each current arrangement of students as per the provided guidelines in the problem.

    -   **Goal State**\
        The goal state is any state where all students are distributed amongst teams in a particular `FringeState`.

    -   **Initial State**\
        The initial state is an empty list, so that the successors can be dynamically generated during the `local search`.

### Highlights for Part 3:
1.  The algorithm used in this problem is an optimized form of greedy search. We are also pruning the search space by rejecting certain states under a cost threshold, where just the low cost states will be moved ahead for further computation.

    The value for the variable `threshold` was obtained by manually running the algorithm with the generated states visited in reverse order.
    

2.  The preferences have been read and stores in a simple Python dictionary, which makes it easy to access each preference information in a quick manner.

3.  The cost for the team is calculated using the function `calculate_relative_total_cost()` which takes into account the perspective of each student, since every student's preference needs to be addressed.

4.  The teams are generated dynamically through iterative checks, where we keep generating new team combinations until all students are distributed across teams.