
## we will maintain an array of two types - endpoints and startpoints
## each endpoint has a reference to the index of a corresponding startpoint
## each startpoint has a reference to the index of the most recent endpoint (accounting for the commutation between different buildings)

## the expected time and space complexity of the algorithm is O(n^2 + np2^n) - n(the number of different period objects), p(the number of different locations)

import sys
import copy
from collections import deque
import time
from functools import wraps

INF = 1e14

## a class encapsulating all the relevant information about a schedule object (period or practical)
class Period:

    DAY = {"Monday":0, "Tuesday":1, "Wednesday":2, "Thursday":3, "Friday":4}
    DynamicDay = {"Monday":0, "Tuesday":1, "Wednesday":2, "Thursday":3, "Friday":4}
    
    def __init__(self, name, location, start, end, day) -> None:
        self.name, self.location, self.start, self.end, self.day = name, location, start, end, day
        self.ordering = Period.convert_to_minutes(start, day, dynamic=False)

    @classmethod
    def convert_to_minutes(cls, t, d, dynamic=True):
        hours, minutes = map(int, t.split(":"))
        if dynamic: return Period.DynamicDay[d]*24*60 + 60*hours + minutes
        return Period.DAY[d]*24*60 + 60*hours + minutes
    
    def __repr__(self) -> str:
        return repr(f"type: \"Period\", period_name: {self.name}, start: {self.start}, end: {self.end}, location: {self.location}, day: {self.day}")
    
    def __lt__(self, other):
        return (self.ordering, self.location)<(other.ordering, other.location)

    def __eq__(self, other):
        return (self.ordering, self.location)==(other.ordering, other.location)
    

class Interval:

    def __init__(self, name, location, val, index) -> None:
        self.period_name = name
        self.location = location
        self.val = val
        self.index = index
        self.sorted_ind = None

    def identity(self):
        pass

    def __lt__(self, other):
        return (self.val, self.location)<(other.val, other.location)

    def __eq__(self, other):
        return (self.val, self.location)==(other.val, other.location)

    def __repr__(self) -> str:
        return repr(f"type: \"Interval\", period_name: {self.period_name}, value: {self.val}, location: {self.location}, index: {self.index}")
        

class EndPoint(Interval):

    def __init__(self, name, location, val, index) -> None:
        super().__init__(name, location, val, index)
        self.start = None


    def identity(self):
        return True
   

class StartPoint(Interval):

    def __init__(self, name, location, val, index) -> None:
        super().__init__(name, location, val, index)
        self.prev_end = None

    def identity(self):
        return False


class SchedulePlanner:

    def __init__(self, dist, mandatory, free_intervals, concentration_where, input_file_name, output_file_name) -> None:

        self.dist = dist
        self.mandatory = mandatory
        self.free_intervals = free_intervals
        self.concentration_where = concentration_where
        self.input_file_name = input_file_name
        self.output_file_name = output_file_name

    def data_proc(self):

        ## construct the array periods

        matchings = [{"Monday":0, "Tuesday":1, "Wednesday":2, "Thursday":3, "Friday":4}, {"Monday":4, "Tuesday":3, "Wednesday":2, "Thursday":1, "Friday":0},
                    {"Monday":4, "Tuesday":2, "Wednesday":0, "Thursday":1, "Friday":3}]

        Period.DynamicDay = matchings[self.concentration_where]

        periods = []

        with open(self.input_file_name, "r") as inp:
            for line in inp.readlines():
                if (line[0] == "#"):
                    continue
                for i in line:
                    if i != " " and i != "\n":
                        break
                else:
                    continue

                l = line.split(); l[1] = int(l[1])
                per = Period(*l)
                if self.is_valid(per): periods.append(per)
        return periods

    def build_schedule(self, periods):
        schedule = []; schedule_end = []; schedule_start = []; schedule_mask = dict()

        ## necessary preprocessing of the sorted array of the starting points and ending points of the periods
        ## populating of the arrays schedule_end, schedule_start, schedule_mask

        for ind, period in enumerate(periods):
            startP = StartPoint(period.name, period.location, Period.convert_to_minutes(period.start, period.day), ind)
            endP = EndPoint(period.name, period.location, Period.convert_to_minutes(period.end, period.day), ind)

            schedule.append(startP); schedule.append(endP)
            schedule_start.append(startP); schedule_end.append(endP)

            if period.name not in schedule_mask:
                schedule_mask[period.name] = len(schedule_mask)
        
        schedule.sort()
        schedule_end.sort()
        last = 0
        for i in schedule:
            if i.identity():
                i.sorted_ind = last; last += 1

        # O(n^2) preprocessing (could be O(n) but why bother)
        for i in range(len(schedule)):
            if (not schedule[i].identity()):
                for j in range(i, -1, -1):
                    if (schedule[j].identity() and schedule[j].val + self.dist[schedule[j].location][schedule[i].location]<=schedule[i].val):
                        schedule[i].prev_end = schedule[j].sorted_ind; break


        return schedule, schedule_start, schedule_end, schedule_mask

    def count_set_bits(self, a):
        cnt = 0
        while(a):
            cnt += 1
            a = a&(a-1)
        return cnt

    def update_best(self, best, candidate, schedule_end):
        if (len(candidate)>len(best) or (len(candidate) == len(best) and schedule_end[candidate[-1]].val<schedule_end[best[-1]].val)):
            return candidate
        else:
            return best

    def mandatory_subjects(self, mask, schedule_mask):
        for i in self.mandatory:
            if i not in schedule_mask: return False
            if (1<<schedule_mask[i])&mask == 0:
                return False
        return True

    def interval_intersect(self, int1, int2):
        start1 = Period.convert_to_minutes(int1[0], int1[2], dynamic=False)
        end1 = Period.convert_to_minutes(int1[1], int1[2], dynamic=False)

        start2 = Period.convert_to_minutes(int2[0], int2[2], dynamic=False)
        end2 = Period.convert_to_minutes(int2[1], int2[2], dynamic=False)

        return not (start1>end2 or start2>end1)

    def is_valid(self, per):
        per_interval = (per.start, per.end, per.day)
        for i in self.free_intervals:
            if self.interval_intersect(per_interval, i):
                return False
        return True
    
    def compile_schedule(self):
        ## array containing the period objects
        periods = self.data_proc(self.concentration_where)

        ## contains the starting and ending points of the periods and practices
        schedule = []

        ## only ending points
        schedule_end = []

        ## only the starting points
        schedule_start = []

        schedule, schedule_start, schedule_end, schedule_mask = self.build_schedule(periods, self.dist)

        condition = lambda sched_obj1, sched_obj2 : sched_obj1.val + self.dist[sched_obj1.location][schedule_start[sched_obj2.index].location]<schedule_start[sched_obj2.index].val

        ## bitmask dynamic programming
        
        LPER = len(periods)
        MASKS = 1<<len(schedule_mask)

        ## array of optimal solutions for each submask of the periods for i-th prefix
        dp_masks = [[[] for j in range(len(self.dist))] for i in range(MASKS)]

        is_active = lambda mask, location : len(dp_masks[mask][location]) == self.count_set_bits(mask)

        best_solution = []

        for i in range(LPER):
            curr_ind = schedule_mask[schedule_end[i].period_name]
            curr_loc = schedule_end[i].location
            chng_bits = (MASKS-1)^(1<<curr_ind)
            mask = 0
            while True:
                mask1 = mask^(1<<curr_ind)
    
                best_dp = []
                for j in range(len(self.dist)):
                    if is_active(mask, j):
                        push_dp = []

                        if (len(dp_masks[mask][j]) == 0):
                            push_dp.append(i)
                        elif condition(schedule_end[dp_masks[mask][j][-1]], schedule_end[i]):
                            push_dp = copy.deepcopy(dp_masks[mask][j])
                            push_dp.append(i)
                        
                        if len(push_dp) > len(best_dp):
                            best_dp = copy.deepcopy(push_dp)
                
                if len(best_dp)>0 and ((len(best_dp) > len(dp_masks[mask1][curr_loc])) or (len(best_dp) == len(dp_masks[mask1][curr_loc]) and \
                    schedule_end[best_dp[-1]].val < schedule_end[dp_masks[mask1][curr_loc][-1]].val)):
                    dp_masks[mask1][curr_loc] = best_dp
                    if self.mandatory_subjects(mask1, schedule_mask):
                        best_solution = self.update_best(best_solution, best_dp, schedule_end)
                    
                mask = (mask-chng_bits)&chng_bits
                if (mask == 0): break


        ## final reconstruction of the optimal solution

        optimal_schedule = []
        for i in best_solution:
            optimal_schedule.append(periods[schedule_end[i].index])
        
        optimal_schedule.sort()

        flag = True

        with open(self.output_file_name, "a") as out:
            if len(optimal_schedule) == 0:
                print("no suitable schedule exists", file=out)
                flag = False
        
            if flag:
                print("_______________________________________________________________________", file=out)
                print(f'the best solution - from the {"start" if self.concentration_where == 0 else "middle" if self.concentration_where == 2 else "end"} of the week', file=out)
                last_day = None
                for i in optimal_schedule:
                    if i.day != last_day:
                        last_day = i.day
                        print("", file=out)
                    print(i, file=out)
            
                for i in range(4):
                    print("", file=out)

    def compile_schedule_all_versions(self):
        for i in range(3):
            self.concentration_where = i
            self.compile_schedule()

def time_duration(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        f(*args, **kwargs)
        end = time.perf_counter()
        print(f"the runtime of the function \"{f.__name__}\" was: ", end-start, "seconds")
    return wrapper


@time_duration
def main():

    ## the matrix of distances between locations (in minutes)
    ## 0 - location 0
    ## 1 - location 1
    ## 2 - location 2

    dist = [[0, 50, 60, 30],
            [50, 0, 70, 35],
            [60, 70, 0, -1],
            [30, 35, -1, 0]]

    ## mandatory subjects (guaranteed to be included in the solution if such solution exists)
    mandatory = ["Programming_prac", "Algorithmization_prac","Programming", "Algorithmization", "Discrete_mathematics", "Discrete_mathematics_prac", 
                 "Computer_networks","Linear_algebra", "Linear_algebra_prac", "Computer_principles", "Python_machine_learning", "Python_machine_learning_prac",
                 "Combinatorics_and_graphs", "Combinatorics_and_graphs_prac", "programming_in_c++", "programming_in_c++_prac"]


    ## time windows that has to remain free during the week (from when, until when, day)
    free_intervals = []

    ## pushing the lessons to the start/end/middle of the week (0, 1, 2, respectively)
    concentration_where = 0

    ## the input file
    input_file_name = "schedule_input.txt"

    ## the output file name
    output_file_name = "schedule_output.txt"

    SchedulePlanner(dist, mandatory, free_intervals, concentration_where, input_file_name, output_file_name).compile_schedule_all_versions()


if __name__ == "__main__":
    main()


## example input data:

# Discrete_mathematics 0 12:20 13:50 Monday
# Discrete_mathematics 1 12:20 13:50 Tuesday
# Discrete_mathematics 0 14:00 15:30 Tuesday
# Computer_networks 0 12:20 13:50 Tuesday
# Computer_networks 1 15:40 17:10 Wednesday
# Computer_networks 0 15:40 17:10 Thursday
# Algorithmization 0 10:40 12:10 Monday
# Algorithmization 1 9:00 10:30 Tuesday
# Algorithmization 0 15:40 17:10 Wednesday
# Programming 1 9:00 10:30 Monday
# Programming 0 10:40 12:10 Tuesday
# Programming 0 17:20 18:50 Wednesday
# Linear_algebra 0 15:40 17:10 Tuesday
# Linear_algebra 1 12:20 13:50 Wednesday
# Linear_algebra 0 10:40 12:10 Friday
# Computer_principles 1 10:40 12:10 Monday
# Computer_principles 0 14:00 15:30 Thursday
# Computer_principles 0 12:20 13:50 Friday
# Python_machine_learning 1 12:20 13:50 Monday
# Python_machine_learning 0 9:00 10:30 Tuesday
# Database_systems 0 17:20 18:50 Monday
# Database_systems 1 9:00 10:30 Thursday
# Combinatorics_and_graphs 1 10:40 12:10 Tuesday
# Combinatorics_and_graphs 0 9:00 10:30 Wednesday
# Statement_a_predicate_logic 0 14:00 15:30 Monday
# Statement_a_predicate_logic 1 14:00 15:30 Tuesday
# Database_systems_prac 1 9:00 10:30 Tuesday
# Database_systems_prac 1 14:00 15:30 Wednesday
# Database_systems_prac 1 15:40 17:10 Wednesday
# Database_systems_prac 1 10:40 12:10 Thursday
# Database_systems_prac 1 12:20 13:50 Thursday
# Database_systems_prac 1 14:00 15:30 Thursday
# Database_systems_prac 1 15:40 17:10 Thursday
# Database_systems_prac 1 9:00 10:30 Friday
# Discrete_mathematics_prac 0 12:20 13:50 Monday
# Discrete_mathematics_prac 0 15:40 17:10 Monday
# Discrete_mathematics_prac 1 15:40 17:10 Monday
# Discrete_mathematics_prac 0 10:40 12:10 Tuesday
# Discrete_mathematics_prac 0 12:20 13:50 Tuesday
# Discrete_mathematics_prac 1 15:40 17:10 Tuesday
# Discrete_mathematics_prac 1 9:00 10:30 Wednesday
# Discrete_mathematics_prac 0 14:00 15:30 Wednesday
# Discrete_mathematics_prac 0 15:40 17:10 Wednesday
# Discrete_mathematics_prac 0 10:40 12:10 Thursday
# Discrete_mathematics_prac 0 12:20 13:50 Thursday
# Discrete_mathematics_prac 0 14:00 15:30 Thursday
# Discrete_mathematics_prac 0 9:00 10:30 Friday
# Discrete_mathematics_prac 0 10:40 12:10 Friday
# Combinatorics_and_graphs_prac 1 15:40 17:10 Monday
# Combinatorics_and_graphs_prac 1 9:00 10:30 Tuesday
# Combinatorics_and_graphs_prac 1 14:00 15:30 Tuesday
# Combinatorics_and_graphs_prac 1 17:20 18:50 Tuesday
# Combinatorics_and_graphs_prac 1 15:40 17:10 Wednesday
# Combinatorics_and_graphs_prac 1 9:00 10:30 Thursday
# Combinatorics_and_graphs_prac 1 10:40 12:10 Thursday
# Combinatorics_and_graphs_prac 1 17:20 18:50 Thursday
# Combinatorics_and_graphs_prac 1 10:40 12:10 Friday
# Programming_prac 0 9:50 10:30 Tuesday
# Algorithmization_prac 0 10:40 12:10 Tuesday
# Python_machine_learning_prac 1 14:00 15:30 Monday
# Python_machine_learning_prac 1 9:00 10:30 Wednesday
# Linear_algebra_prac 0 12:20 13:50 Monday
# Linear_algebra_prac 0 14:00 15:30 Monday
# Linear_algebra_prac 0 9:00 10:30 Tuesday
# Linear_algebra_prac 1 10:40 12:10 Tuesday
# Linear_algebra_prac 1 14:00 15:30 Tuesday
# Linear_algebra_prac 0 17:20 18:50 Tuesday
# Linear_algebra_prac 1 9:00 10:30 Wednesday
# Linear_algebra_prac 0 14:00 15:30 Wednesday
# Linear_algebra_prac 0 15:40 17:10 Wednesday
# Linear_algebra_prac 0 17:20 18:50 Wednesday
# Linear_algebra_prac 0 9:00 10:30 Thursday
# Linear_algebra_prac 0 10:40 12:10 Thursday
# Linear_algebra_prac 0 12:20 13:50 Thursday
# Linear_algebra_prac 0 14:00 15:30 Thursday
# Linear_algebra_prac 0 9:00 10:30 Friday
# Linear_algebra_prac 0 10:40 12:10 Friday
# Linear_algebra_prac 0 14:00 15:30 Friday
# Statement_a_predicate_logic_prac 1 15:40 17:10 Tuesday
# Statement_a_predicate_logic_prac 1 14:00 15:30 Wednesday
# Statement_a_predicate_logic_prac 1 9:00 10:30 Thursday
# Statement_a_predicate_logic_prac 1 10:40 12:10 Thursday
# Statement_a_predicate_logic_prac 1 12:20 13:50 Thursday
# Statement_a_predicate_logic_prac 1 9:00 10:30 Friday
# Statement_a_predicate_logic_prac 1 10:40 12:10 Friday
# Statement_a_predicate_logic_prac 1 12:20 13:50 Friday
# programming_in_c++ 0 10:40 12:10 Wednesday
# programming_in_c++ 1 14:00 15:30 Wednesday
# programming_in_c++_prac 0 12:20 13:50 Monday
# programming_in_c++_prac 1 12:20 13:50 Monday
# programming_in_c++_prac 1 12:20 13:50 Tuesday
# programming_in_c++_prac 1 14:00 15:30 Wednesday
# programming_in_c++_prac 1 17:20 18:50 Wednesday
# programming_in_c++_prac 1 12:20 13:50 Thursday
# programming_in_c++_prac 1 15:40 17:10 Thursday
# programming_in_c++_prac 1 10:40 12:10 Friday
# programming_in_c++_prac 1 12:20 13:50 Friday
