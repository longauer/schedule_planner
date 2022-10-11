
## we will maintain an array of two types - endpoints and startpoints
## each endpoint has a reference to the index of a corresponding startpoint
## each startpoint has a reference to the index of the most recent endpoint (accounting for the commutation between different buildings)

## the expected time and space complexity of the algorithm is O(n2^n + n^2) 

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
    def convert_to_minutes(cls ,t, d, dynamic=True):
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



def data_proc(concentration_where, free_intervals):

    ## construct the array periods

    matchings = [{"Monday":0, "Tuesday":1, "Wednesday":2, "Thursday":3, "Friday":4}, {"Monday":4, "Tuesday":3, "Wednesday":2, "Thursday":1, "Friday":0},
                 {"Monday":4, "Tuesday":2, "Wednesday":0, "Thursday":1, "Friday":3}]

    Period.DynamicDay = matchings[concentration_where]

    periods = []

    with open("schedule_input.txt", "r") as inp:
        for line in inp.readlines():
            l = line.split(); l[1] = int(l[1])
            per = Period(*l)
            if is_valid(per, free_intervals): periods.append(per)
    return periods

def build_schedule(periods, dist):
    schedule = []; schedule_end = []; schedule_start = []; schedule_mask = dict()

    ## do the necessary preprocessing of the sorted array of the starting points and ending points of the periods
    ## populate the arrays schedule_end, schedule_start, schedule_mask

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
                if (schedule[j].identity() and schedule[j].val + dist[schedule[j].location][schedule[i].location]<=schedule[i].val):
                    schedule[i].prev_end = schedule[j].sorted_ind; break


    return schedule, schedule_start, schedule_end, schedule_mask

def count_set_bits(a):
    cnt = 0
    while(a):
        cnt += 1
        a = a&(a-1)
    return cnt

def update_best(best, candidate, schedule_end):
    if (len(candidate)>len(best) or (len(candidate) == len(best) and schedule_end[candidate[-1]].val<schedule_end[best[-1]].val)):
        return candidate
    else:
        return best

def mandatory_subjects(mask, mandatory, schedule_mask):
    for i in mandatory:
        if i not in schedule_mask: return False
        if (1<<schedule_mask[i])&mask == 0:
            return False
    return True

def interval_intersect(int1, int2):
    start1 = Period.convert_to_minutes(int1[0], int1[2], dynamic=False)
    end1 = Period.convert_to_minutes(int1[1], int1[2], dynamic=False)

    start2 = Period.convert_to_minutes(int2[0], int2[2], dynamic=False)
    end2 = Period.convert_to_minutes(int2[1], int2[2], dynamic=False)

    return not (start1>end2 or start2>end1)

def is_valid(per, free_intervals):
    per_interval = (per.start, per.end, per.day)
    for i in free_intervals:
        if interval_intersect(per_interval, i):
            return False
    return True

def time_duration(f):
    @wraps(f)
    def wrapper():
        start = time.perf_counter()
        f()
        end = time.perf_counter()
        print(f"the runtime of the function \"{f.__name__}\" was: ", end-start, "seconds")
    return wrapper


@time_duration
def main():

    ## the matrix of distances between locations
    dist = [[0, 50],
            [50, 0]]

    ## mandatory subjects (guaranteed to be included in the solution if such solution exists)
    mandatory = ["Programovani1_cvic", "Algoritmizace_cvic","Programovani1", "Algoritmizace", "Diskretni_matematika", "Diskretni_matematika_cvic", 
                 "Uvod_do_pocitacovych_siti","Linearni_algebra", "Linearni_algebra_cvic", "Principy_pocitacu", "Python_machine_learning", "Python_machine_learning_cvic",
                 "Kombinatorika_a_grafy", "Kombinatorika_a_grafy_cvic", "programovani_v_c++", "programovani_v_c++_cvic"]

    ## time windows that has to remain free during the week (from when, until when, day)
    free_intervals = [] # add the implementation

    ## pushing the lessons to the start/end/middle of the week (0, 1, 2, respectively)
    concentration_where = 0

    ## array containing the period objects
    periods = data_proc(concentration_where, free_intervals)

    ## contains the starting and ending points of the periods and practices
    schedule = []

    ## only ending points
    schedule_end = []

    ## only the starting points
    schedule_start = []

    schedule, schedule_start, schedule_end, schedule_mask = build_schedule(periods, dist)

    condition = lambda sched_obj1, sched_obj2 : sched_obj1.val + dist[sched_obj1.location][schedule_start[sched_obj2.index].location]<schedule_start[sched_obj2.index].val

    ## bitmask dp
    
    LPER = len(periods)
    MASKS = 1<<len(schedule_mask)

    ## array of optimal solutions for each submask of the periods for i-th prefix
    dp_masks = [[] for i in range(MASKS)]

    is_active = lambda mask : len(dp_masks[mask]) == count_set_bits(mask)

    best_solution = []
                
    for i in range(LPER):
        curr_ind = schedule_mask[schedule_end[i].period_name]
        chng_bits = (MASKS-1)^(1<<curr_ind)
        mask = 0
        while True:
            mask1 = mask^(1<<curr_ind)
            if is_active(mask) and not is_active(mask1):
                push_dp = []
                flag = True
                if (len(dp_masks[mask]) == 0):
                    push_dp.append(i)
                elif condition(schedule_end[dp_masks[mask][-1]], schedule_end[i]):
                    push_dp = copy.deepcopy(dp_masks[mask])
                    push_dp.append(i)
                else:
                    flag = False
                
                if (flag):
                    dp_masks[mask1] = push_dp
                    if mandatory_subjects(mask1, mandatory, schedule_mask):
                        best_solution = update_best(best_solution, push_dp, schedule_end)
            mask = (mask-chng_bits)&chng_bits
            if (mask == 0): break

    

    ## final reconstruction of the optimal solution

    optimal_schedule = []
    for i in best_solution:
        optimal_schedule.append(periods[schedule_end[i].index])
    
    optimal_schedule.sort()


    with open("schedule_output.txt", "a") as out:
        if len(optimal_schedule) == 0:
            print("no suitable schedule exists", file=out)
        print(file=out)
        for i in optimal_schedule:
            print(i, file=out)


if __name__ == "__main__":
    main()


## the periods

# Diskretni_matematika 0 12:20 13:50 Monday
# Diskretni_matematika 1 12:20 13:50 Tuesday
# Diskretni_matematika 0 14:00 15:30 Tuesday
# Uvod_do_pocitacovych_siti 0 12:20 13:50 Tuesday
# Uvod_do_pocitacovych_siti 1 15:40 17:10 Wednesday
# Uvod_do_pocitacovych_siti 0 15:40 17:10 Thursday
# Algoritmizace 0 10:40 12:10 Monday
# Algoritmizace 1 9:00 10:30 Tuesday
# Algoritmizace 0 15:40 17:10 Wednesday
# Programovani1 1 9:00 10:30 Monday
# Programovani1 0 10:40 12:10 Tuesday
# Programovani1 0 17:20 18:50 Wednesday
# Linearni_algebra 0 15:40 17:10 Tuesday
# Linearni_algebra 1 12:20 13:50 Wednesday
# Linearni_algebra 0 10:40 12:10 Friday
# Principy_pocitacu 1 10:40 12:10 Monday
# Principy_pocitacu 0 14:00 15:30 Thursday
# Principy_pocitacu 0 12:20 13:50 Friday
# Python_machine_learning 1 12:20 13:50 Monday
# Python_machine_learning 0 9:00 10:30 Tuesday
# Databazove_systemy 0 17:20 18:50 Monday
# Databazove_systemy 1 9:00 10:30 Thursday
# Kombinatorika_a_grafy 1 10:40 12:10 Tuesday
# Kombinatorika_a_grafy 0 9:00 10:30 Wednesday
# Vyrokova_a_predikatova_logika 0 14:00 15:30 Monday
# Vyrokova_a_predikatova_logika 1 14:00 15:30 Tuesday
# Databazove_systemy_cvic 1 9:00 10:30 Tuesday
# Databazove_systemy_cvic 1 14:00 15:30 Wednesday
# Databazove_systemy_cvic 1 15:40 17:10 Wednesday
# Databazove_systemy_cvic 1 10:40 12:10 Thursday
# Databazove_systemy_cvic 1 12:20 13:50 Thursday
# Databazove_systemy_cvic 1 14:00 15:30 Thursday
# Databazove_systemy_cvic 1 15:40 17:10 Thursday
# Databazove_systemy_cvic 1 9:00 10:30 Friday
# Diskretni_matematika_cvic 0 12:20 13:50 Monday
# Diskretni_matematika_cvic 0 15:40 17:10 Monday
# Diskretni_matematika_cvic 1 15:40 17:10 Monday
# Diskretni_matematika_cvic 0 10:40 12:10 Tuesday
# Diskretni_matematika_cvic 0 12:20 13:50 Tuesday
# Diskretni_matematika_cvic 1 15:40 17:10 Tuesday
# Diskretni_matematika_cvic 1 9:00 10:30 Wednesday
# Diskretni_matematika_cvic 0 14:00 15:30 Wednesday
# Diskretni_matematika_cvic 0 15:40 17:10 Wednesday
# Diskretni_matematika_cvic 0 10:40 12:10 Thursday
# Diskretni_matematika_cvic 0 12:20 13:50 Thursday
# Diskretni_matematika_cvic 0 14:00 15:30 Thursday
# Diskretni_matematika_cvic 0 9:00 10:30 Friday
# Diskretni_matematika_cvic 0 10:40 12:10 Friday
# Kombinatorika_a_grafy_cvic 1 15:40 17:10 Monday
# Kombinatorika_a_grafy_cvic 1 9:00 10:30 Tuesday
# Kombinatorika_a_grafy_cvic 1 14:00 15:30 Tuesday
# Kombinatorika_a_grafy_cvic 1 17:20 18:50 Tuesday
# Kombinatorika_a_grafy_cvic 1 15:40 17:10 Wednesday
# Kombinatorika_a_grafy_cvic 1 9:00 10:30 Thursday
# Kombinatorika_a_grafy_cvic 1 10:40 12:10 Thursday
# Kombinatorika_a_grafy_cvic 1 17:20 18:50 Thursday
# Kombinatorika_a_grafy_cvic 1 10:40 12:10 Friday
# Programovani1_cvic 0 9:50 10:30 Tuesday
# Algoritmizace_cvic 0 10:40 12:10 Tuesday
# Python_machine_learning_cvic 1 14:00 15:30 Monday
# Python_machine_learning_cvic 1 9:00 10:30 Wednesday
# Linearni_algebra_cvic 0 12:20 13:50 Monday
# Linearni_algebra_cvic 0 14:00 15:30 Monday
# Linearni_algebra_cvic 0 9:00 10:30 Tuesday
# Linearni_algebra_cvic 1 10:40 12:10 Tuesday
# Linearni_algebra_cvic 1 14:00 15:30 Tuesday
# Linearni_algebra_cvic 0 17:20 18:50 Tuesday
# Linearni_algebra_cvic 1 9:00 10:30 Wednesday
# Linearni_algebra_cvic 0 14:00 15:30 Wednesday
# Linearni_algebra_cvic 0 15:40 17:10 Wednesday
# Linearni_algebra_cvic 0 17:20 18:50 Wednesday
# Linearni_algebra_cvic 0 9:00 10:30 Thursday
# Linearni_algebra_cvic 0 10:40 12:10 Thursday
# Linearni_algebra_cvic 0 12:20 13:50 Thursday
# Linearni_algebra_cvic 0 14:00 15:30 Thursday
# Linearni_algebra_cvic 0 9:00 10:30 Friday
# Linearni_algebra_cvic 0 10:40 12:10 Friday
# Linearni_algebra_cvic 0 14:00 15:30 Friday
# Vyrokova_a_predikatova_logika_cvic 1 15:40 17:10 Tuesday
# Vyrokova_a_predikatova_logika_cvic 1 14:00 15:30 Wednesday
# Vyrokova_a_predikatova_logika_cvic 1 9:00 10:30 Thursday
# Vyrokova_a_predikatova_logika_cvic 1 10:40 12:10 Thursday
# Vyrokova_a_predikatova_logika_cvic 1 12:20 13:50 Thursday
# Vyrokova_a_predikatova_logika_cvic 1 9:00 10:30 Friday
# Vyrokova_a_predikatova_logika_cvic 1 10:40 12:10 Friday
# Vyrokova_a_predikatova_logika_cvic 1 12:20 13:50 Friday
# programovani_v_c++ 0 10:40 12:10 Wednesday
# programovani_v_c++ 1 14:00 15:30 Wednesday
# programovani_v_c++_cvic 0 12:20 13:50 Monday
# programovani_v_c++_cvic 1 12:20 13:50 Monday
# programovani_v_c++_cvic 1 12:20 13:50 Tuesday
# programovani_v_c++_cvic 1 14:00 15:30 Wednesday
# programovani_v_c++_cvic 1 17:20 18:50 Wednesday
# programovani_v_c++_cvic 1 12:20 13:50 Thursday
# programovani_v_c++_cvic 1 15:40 17:10 Thursday
# programovani_v_c++_cvic 1 10:40 12:10 Friday
# programovani_v_c++_cvic 1 12:20 13:50 Friday
