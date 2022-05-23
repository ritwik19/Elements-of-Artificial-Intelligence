#!/usr/local/bin/python3
# route.py : Find routes through maps
#
# Code by: Harsh Srivastava <hsrivas>
#        : Ritwik Budhiraja <rbudhira>
#        : Yash Kalpesh Shah <yashah>
#
# Based on skeleton code by V. Mathur and D. Crandall, January 2021
#


# !/usr/bin/env python3
import heapq
import math
import sys

# Function to read City GPS information
def read_city_gps():
    """
    This function reads the text based city-gps data and stores it as an in-memory Python dictionary.

    The location of the city is read as degrees but stores in radians to ease later calculations.
    """
    raw_gps_data = open("./city-gps.txt", "r").readlines()
    raw_gps_data = [line.split() for line in raw_gps_data]
    raw_gps_data = [(line_split[0], (math.radians(float(line_split[1])), math.radians(
        float(line_split[2])))) for line_split in raw_gps_data]
    return dict(raw_gps_data)

# Function to read Road Segment information
#   - This function reads each from-to city pairs and saves a double mapping for two way road lookups


def read_road_segments():
    """
    @brief
    This function reads the text based road segment information from the road-segments.txt.

    @desc
    The data is stored in the form of a nested Python dictionary. The from city/junction is
        the key for the top level dictionary. Followed by the second level storing the data
        for destinations connected through the from city/junction.

    The function also calculates the maximum_speed limit and maximum_segment_length (max
        segment distance) from the data provided for later use.

    @returns segment data, maximum speed limit and maximum segment length.
    """
    maximum_speed = 0
    maximum_segment_length = 0

    raw_road_segments = open("./road-segments.txt", "r").readlines()
    raw_road_segments = [line.split() for line in raw_road_segments]
    road_segments = {
        # "Bloomginton,_Indiana": {
        #     "New_York,_New_York": (100, 150, "R9"),
        #     "Newark,_New_Jersey": (90, 175, "R9")
        # }
    }

    # Helper inner function to add (start, end) city pair to the nested dictionary
    def conditionally_add_segment_info(start, end, data):
        if start not in road_segments:
            road_segments[start] = {}
        road_segments[start][end] = data

    # Process the raw data
    for line_split in raw_road_segments:
        distance = float(line_split[2])
        speed_limit = float(line_split[3])
        # Form the data tuple
        data = (
            # Distance
            distance,
            # Speed Limit
            speed_limit,
            # Highway Name
            line_split[4]
        )

        maximum_speed = max(maximum_speed, speed_limit)
        maximum_segment_length = max(maximum_segment_length, distance)

        # Add (from, to) -> data mapping
        conditionally_add_segment_info(line_split[0], line_split[1], data)

        # Add (to, from) -> data mapping
        conditionally_add_segment_info(line_split[1], line_split[0], data)
    return road_segments, maximum_speed, maximum_segment_length


class RouteState:
    """
    Represents any route taken during the whole search process.
    The class is comprised of all useful data from distance traveled to time taken, which can be used to calculate various heuristics.

    Members:
        origin (str) - Current city for the state
        route (list) - The route taken to reach this state of computation
        segment_length (int) - Distance of the previous segment taken to the origin city
        segment_count (int) - Count the segments already travelled so far to reach this state
        traveled_distance (int) - The distance travelled so far
        time_taken (float) - The time taken to travel the distance
    """

    def __init__(self, origin: str,
                    previous_route: list = [],
                    segment_length=0,
                    name=None,
                    total_previous_traveled_distance: int = 0,
                    total_previous_time_taken: int = 0,
                    total_previous_time_taken_delivery: int = 0,
                    speed_limit=0,
                    maximum_speed_limit=0,
                    parent=None,
                    g_previous=0.0,
                    g_heuristic_function=None,
                    h_heuristic_function=None):
        self.origin = origin
        self.route = previous_route + [(self.origin, segment_length, name)]

        self.segment_length = segment_length

        # Calculate the segment count, one less than total route length
        self.segment_count = len(self.route) - 1

        self.speed_limit = speed_limit

        # Self update the maximum speed limit through the route
        self.maximum_speed_limit = max(maximum_speed_limit, self.speed_limit)

        self.time_taken = ((self.segment_length / self.speed_limit) if self.speed_limit > 0 else 0.0)
        self.time_taken_delivery = self.time_taken + math.tanh(self.segment_length / 1000.0) * 2.0 * (self.time_taken + total_previous_time_taken_delivery)

        self.total_traveled_distance = total_previous_traveled_distance + self.segment_length
        self.total_time_taken = total_previous_time_taken + self.time_taken
        self.total_time_delivery = total_previous_time_taken_delivery + self.time_taken_delivery

        self.name = name
        self.parent = parent

        self.g_previous = g_previous

        self.g_function = g_heuristic_function
        if self.g_function == None:
            self.g_function = lambda x: len(self.route) - 1

        self.h_function = h_heuristic_function
        if self.h_function == None:
            self.h_function = lambda x: len(self.route) - 1

        self.g = self.g_previous + self.g_function(self)
        self.h = self.h_function(self)
        self.f = self.g + self.h

        if self.f == math.inf:
            self.hash = sys.maxsize
        else:
            self.hash = int(round(self.f, 8) * 100000000.0)

    def successors(self, gps_data, segment_data):
        """
        This is the successor function.

        Returns: Possible connections of cities that can be reached from the origin
        """
        # segment_data["Bloomginton,_Indiana"]
        #   => {
        #          "New_York,_New_York": (100, 150, "R9"),
        #          "Newark,_New_Jersey": (90, 175, "R8")
        #      }
        successors_list = []

        if self.origin in segment_data:
            for (city, data) in segment_data[self.origin].items():
                successors_list.append(
                    RouteState(city,
                            self.route,
                            segment_length=data[0],
                            name=data[2],
                            total_previous_traveled_distance=self.total_traveled_distance,
                            total_previous_time_taken=self.total_time_taken,
                            total_previous_time_taken_delivery=self.total_time_delivery,
                            speed_limit=data[1],
                            maximum_speed_limit=self.maximum_speed_limit,
                            parent=self,
                            g_previous=self.g,
                            g_heuristic_function=self.g_function,
                            h_heuristic_function=self.h_function)
                )
        return successors_list

    def __lt__(self, value):
        return self.f < value.f

    def __str__(self):
        return str((self.origin, self.g))

    def __repr__(self):
        return str((self.origin, self.g))

    def __hash__(self):
        return self.hash

    def __eq__(self, value):
        return self.__hash__() == value.__hash__() \
            and self.origin == value.origin
            # and self.segment_count == value.segment_count

    def dump(self):
        return self.__dict__


def resolve_junction(gps_data, segment_data, jct):
    # print("Resolving", jct, "\n\tSegment data", segment_data[jct])

    def find_nearby_cities(from_city, level=0, previous_real_cities=[], previous_junctions=[]):
        # print("\t" * level, "\033[41;37;1m find nearby city \033[0m for ", from_city)
        # Find the list of junctions ending with one real city
        jct_connections = segment_data[from_city]
        real_cities = list(filter(lambda x: not x.startswith("Jct_") and x not in previous_real_cities, jct_connections))
        # print("\t" * level, "\033[46;37;1m real cities      \033[0m    =", real_cities)
        real_cities_props = [[(city, segment_data[from_city][city])] for city in set(real_cities)]

        if level < 1: # and len(real_cities_props) < 3:
            junctions = list(filter(lambda x: x.startswith("Jct_") and x not in previous_junctions , jct_connections))
            for junction in junctions:
                junction_nearby_cities = find_nearby_cities(junction, level + 1, (real_cities + previous_real_cities), (junctions + previous_junctions) + [from_city])
                junction_nearby_cities = list(filter(lambda x: not x[0][0].startswith("Jct_") and x[0][0] not in (real_cities + previous_real_cities), junction_nearby_cities))
                # print("\t\033[31mUn-Expanded Path\033[0m", junction_nearby_cities)
                junction_nearby_cities_temp = []
                for path in junction_nearby_cities:
                    path.append((junction, segment_data[from_city][junction]))
                    junction_nearby_cities_temp.append(path)
                # print("\t\033[31mExpanded Path\033[0m", junction_nearby_cities)
                real_cities_props += junction_nearby_cities

        return real_cities_props

    numerator = [0.0, 0.0]
    denominator = 1.0

    def divide_segments_minimize_gap(path):
        def transform_path_to_dist(sub_path):
            for x in sub_path:
                yield x[1][0]

        def get_new_minimum(i):
            return sum(transform_path_to_dist(path[:i])) - sum(transform_path_to_dist(path[i:]))

        i = len(path) // 2
        last = math.inf
        minimum = get_new_minimum(i)
        while abs(minimum) < last:
            last = minimum
            if i < 0:
                i = i + 1
            minimum = get_new_minimum(i)
        return abs(minimum)

    def divide_segments_maximize_gap(path):
        return sum([x[1][0] for x in path])

    for (to, data) in segment_data[jct].items():
        if to.startswith("Jct_"):
            # pass
            nearby_city_paths = find_nearby_cities(to)
            for nearby_city_path in nearby_city_paths:
                last_gps = gps_data[nearby_city_path[0][0]]
                minimized_distanceto_city = divide_segments_minimize_gap(nearby_city_path)
                maximized_distanceto_city = divide_segments_maximize_gap(nearby_city_path)
                latitude_minimized = minimized_distanceto_city * last_gps[0]
                latitude_maximized = maximized_distanceto_city * last_gps[0]
                longitude_minimized = minimized_distanceto_city * last_gps[1]
                longitude_maximized = maximized_distanceto_city * last_gps[1]
                numerator[0] = (latitude_minimized + latitude_maximized) / 2
                numerator[1] = (longitude_minimized + longitude_maximized) / 2
                denominator += (maximized_distanceto_city + minimized_distanceto_city) / 2
        else:
            if to not in gps_data:
                continue
            to_gps = gps_data[to]
            numerator[0] += data[1] * to_gps[0]
            numerator[1] += data[1] * to_gps[1]
            denominator += data[1]

    gps_data[jct] = (numerator[0] / denominator, numerator[1] / denominator)
    # print("\033[41;37;1m Found \033[0m: GPS(", jct, ") = ", gps_data[jct])


def get_route(start, end, cost):
    """
    Find shortest driving route between start city and end city
    based on a cost function.

    1. Your function should return a dictionary having the following keys:
        -"route-taken" : a list of pairs of the form (next-stop, segment-info), where
           next-stop is a string giving the next stop in the route, and segment-info is a free-form
           string containing information about the segment that will be displayed to the user.
           (segment-info is not inspected by the automatic testing program).
        -"total-segments": an integer indicating number of segments in the route-taken
        -"total-miles": a float indicating total number of miles in the route-taken
        -"total-hours": a float indicating total amount of time in the route-taken
        -"total-delivery-hours": a float indicating the expected (average) time 
                                   it will take a delivery driver who may need to return to get a new package
    2. Do not add any extra parameters to the get_route() function, or it will break our grading and testing code.
    3. Please do not use any global variables, as it may cause the testing code to fail.
    4. You can assume that all test cases will be solvable.
    5. The current code just returns a dummy solution.
    """

    gps_data = read_city_gps()
    segment_data, maximum_speed, maximum_segment_length = read_road_segments()

    no_result = {
        "total-segments": 0,
        "total-miles": 0.0,
        "total-hours": 0.0,
        "total-delivery-hours": 0.0,
        "route-taken": []
    }

    if (start not in segment_data) or (end not in segment_data):
        return no_result

    def cost_segments(state: RouteState):
        return 1

    def cost_distance(state: RouteState):
        return state.segment_length

    def cost_time(state: RouteState):
        return state.time_taken

    def cost_delivery(state: RouteState):
        return state.time_taken_delivery

    # Reference: The idea for calculating the haversine distance has been adopted from the URL given below:
    #    https://www.kite.com/python/answers/how-to-find-the-distance-between-two-lat-long-coordinates-in-python

    def calculate_distance_haversine(point1, point2):
            diff_lat = point1[0] - point2[0]
            diff_lon = point1[1] - point2[1]
            a = (math.sin(diff_lat / 2) ** 2) + \
                math.cos(point1[0]) * math.cos(point2[0]) * \
                (math.sin(diff_lon / 2) ** 2)
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            return 3958.8 * c

    def heuristic_segments(state: RouteState):
        try:
            point1 = gps_data[state.origin]
            point2 = gps_data[end]
            return calculate_distance_haversine(point1, point2) / maximum_segment_length
        except KeyError as e:
            error_key = e.args[0]
            if error_key != None and error_key.startswith("Jct_"):
                # return 0
                resolve_junction(gps_data, segment_data, error_key)
                return heuristic_segments(state)
            else:
                return math.inf

    def heuristic_distance(state: RouteState):
        # Calculate the Haversine distance between goal state and current state
        try:
            point1 = gps_data[state.origin]
            point2 = gps_data[end]
            return calculate_distance_haversine(point1, point2)
        except KeyError as e:
            error_key = e.args[0]
            if error_key != None and error_key.startswith("Jct_"):
                return 0
                # resolve_junction(gps_data, segment_data, error_key)
                # return heuristic_distance(state)
                # return state.parent.total_traveled_distance - state.segment_length
            else:
                return math.inf

    def heuristic_time(state: RouteState):
        # Calculate the Haversine distance between goal state and current state
        try:
            # if state.parent == None:
            point1 = gps_data[state.origin]
            point2 = gps_data[end]
            if state.maximum_speed_limit == 0:
                return calculate_distance_haversine(point1, point2) / maximum_speed
                # return state.time_taken
            return calculate_distance_haversine(point1, point2) / state.maximum_speed_limit
            # else:
            #     return state.parent.g - state.time_taken
        except KeyError as e:
            error_key = e.args[0]
            if error_key != None and error_key.startswith("Jct_"):
                return 0
                resolve_junction(gps_data, segment_data, error_key)
                return heuristic_time(state)
                # return state.parent.total_time_taken - state.time_taken
            else:
                return math.inf

    def heuristic_delivery(state: RouteState):
        # Calculate the Haversine distance between goal state and current state
        try:
            # if state.origin.startswith("Jct_"):
            #     return 0
            if state.parent == None:
                point1 = gps_data[state.origin]
                point2 = gps_data[end]
                return (calculate_distance_haversine(point1, point2) / maximum_speed) - state.time_taken
            else:
                point1 = gps_data[state.parent.origin]
                point2 = gps_data[end]
                t_trip = calculate_distance_haversine(point1, point2) / state.maximum_speed_limit
                t_road = state.time_taken
                h = t_road
                if state.speed_limit >= 50.0:
                    p = math.tanh(state.segment_length / 1000.0)
                    h += p * 2 * (t_road + t_trip)
                return h + state.total_time_taken
        except KeyError as e:
            error_key = e.args[0]
            if error_key != None and error_key.startswith("Jct_"):
                resolve_junction(gps_data, segment_data, error_key)
                return heuristic_delivery(state)
                # return state.parent.total_time_taken - state.time_taken
                return 0
            else:
                return math.inf

    cost_function = None
    heuristic_function = heuristic_distance
    if cost == "segments":
        cost_function = cost_segments
        heuristic_function = heuristic_segments
    elif cost == "distance":
        cost_function = cost_distance
        heuristic_function = heuristic_distance
    elif cost == "time":
        cost_function = cost_time
        heuristic_function = heuristic_time
    elif cost == "delivery":
        cost_function = cost_delivery
        heuristic_function = heuristic_delivery
    else:
        raise ValueError(
            "Invalid value of cost function specified, required: (segments, distance, time, delivery)")

    initial_state = RouteState(start,
                               g_heuristic_function=cost_function,
                               h_heuristic_function=heuristic_function)
    fringe = [initial_state]
    visited = set()

    while len(fringe) > 0:
        current_state: RouteState = heapq.heappop(fringe)

        # Check if goal city has been reached
        if current_state.origin == end:
            # Return the route of the current city
            current_state.route = [(x[0], "{} for {} miles".format(x[2], int(x[1])))
                     for x in current_state.route[1:]]
            return {
                "total-segments": len(current_state.route),
                "total-miles": current_state.total_traveled_distance,
                "total-hours": current_state.total_time_taken,
                "total-delivery-hours": current_state.total_time_delivery,
                "route-taken": current_state.route
            }

        for successor in current_state.successors(gps_data, segment_data):
            # print("\033[42;37;1m Adding \033[0m", successor.origin)
            if successor not in fringe:
                heapq.heappush(fringe, successor)

        # Heapify
        heapq.heapify(fringe)

    return no_result


# Please don't modify anything below this line
#
if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise(Exception("Error: expected 3 arguments"))

    (_, start_city, end_city, cost_function) = sys.argv
    if cost_function not in ("segments", "distance", "time", "delivery"):
        raise(Exception("Error: invalid cost function"))

    result = get_route(start_city, end_city, cost_function)

    # Pretty print the route
    print("Start in %s" % start_city)
    for step in result["route-taken"]:
        print("   Then go to %s via %s" % step)

    print("\n          Total segments: %4d" % result["total-segments"])
    print("             Total miles: %8.3f" % result["total-miles"])
    print("             Total hours: %8.3f" % result["total-hours"])
    print("Total hours for delivery: %8.3f" % result["total-delivery-hours"])
