
## we will maintain an array of two types - endpoints and startpoints
## each endpoint has a reference to the index of a corresponding startpoint
## each startpoint has a reference to the index of the most recent endpoint (accounting for the commutation between different buildings)

## the expected time and space complexity of the algorithm is O(n^2 + np2^n) - n(the number of different period objects), p(the number of different locations)

import sys
import copy
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
                    {"Monday":4, "Tuesday":2, "Wednesday":0, "Thursday":1, "Friday":3}, {"Monday":4, "Tuesday":1, "Wednesday":0, "Thursday":2, "Friday":3}]

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
        periods = self.data_proc()

        ## contains the starting and ending points of the periods and practices
        schedule = []

        ## only ending points
        schedule_end = []

        ## only the starting points
        schedule_start = []

        schedule, schedule_start, schedule_end, schedule_mask = self.build_schedule(periods)

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
                print(f'the best solution - from the {"start" if self.concentration_where == 0 else "middle 1" if self.concentration_where == 2 else "middle 2" if self.concentration_where == 3 else "end"} of the week', file=out)
                last_day = None
                for i in optimal_schedule:
                    if i.day != last_day:
                        last_day = i.day
                        print("", file=out)
                    print(i, file=out)
            
                for i in range(4):
                    print("", file=out)

    def compile_schedule_all_versions(self):
        for i in range(4):
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


def load_mandatory_subjects(filename : str):
    res = list()
    with open (filename, "r") as file:
        for line in file.readlines():
            res.append(line.strip())
    return res

def load_dist_matrix(filename : str):
    matrix = []
    ## first find out the dimension of the square matrix
    with open(filename, "r") as file:
        line = file.readline()
        line = [int(x) for x in line.split()]
        
        matrix.append(line)
        for i in range(len(line)-1):
            line = file.readline()
            line = [int(x) for x in line.split()]
            matrix.append(line)
    return matrix

def load_free_intervals(filename : str):
    res = list()
    with open (filename, "r") as file:
        for line in file.readlines():
            res.append(tuple([x for x in line.strip().split()]))
    return res

@time_duration
def main():
    config_file_name = sys.argv[1]

    config = open(config_file_name, "r")
    file_names = config.read().split()
    print("filenames:\n")
    for i in file_names:
        print(i)

    config.close()
    ## the matrix of distances between locations (in minutes) example
    
    ## loading the distance matrix
    dist = load_dist_matrix(file_names[0])


    ## mandatory subjects (guaranteed to be included in the solution if such solution exists)
    mandatory = load_mandatory_subjects(file_names[1])


    ## time windows that has to remain free during the week (from when, until when, day)
    free_intervals = load_free_intervals(file_names[2])

    ## pushing the lessons to the start/end/middle of the week (0, 1, 2, respectively)
    concentration_where = 0


    ## the input file 
    input_file_name = file_names[3]

    ## the output file name
    output_file_name = file_names[4]

    SchedulePlanner(dist, mandatory, free_intervals, concentration_where, input_file_name, output_file_name).compile_schedule_all_versions()


if __name__ == "__main__":
    main()
