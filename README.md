# schedule_planner

The purpose of the program "schedule_bitmask_dp.py" is to help with organization and optimization of your schedules. Considering the context and the motivation for this project, the optimality of a schedule is defined as follows: the largest possible number of different types of lessons of your choice is included in the schedule, the tie-breaker being the total number of consecutive days appended to your weekend. The program allows for a level of customization in the following: searching for the schedules with maximized number of vacant days, hours and minutes towards the end/beginning/both the end and the beginning of the working week (among these three you will find your optimal schedule), you can search for schedules with certain free time windows being enforced during any given day e.g. for lunch breaks, you can choose a set of lessons to be mandatory - meaning it will be certainly included in the schedule provided such schedule exists. Considering the time complexity function of the algorithm used and the overall implementation, you are not as constricted by the number of different possible dates of the same lessons as you are by the number of different lessons in the input. For this reason, it is not advised you include more than 24 different types of lessons in your input, for program to terminate in a reasonable amount of time. After each time the program finishes the runtime will be displayed in the terminal, which allows for experimentation with varying input sizes on your local machine.

The program accounts for the possibility of your lessons taking place in different places in different times during the week. Therefore, it is required that you fill in the distance matrix "dist" with the size of the square matrix corresponding to the number of all different locations of the lessons. Every lesson is represented by some index (0 - (num_of_locations - 1)) in the matrix and identically later in the input file. Each number in the "dist" then represents the time in minutes for you to get from any starting location to any other.

The input is taken from the text file "schedule_input.txt".
The output is displayed in the text file "schedule_output.txt".

Make sure the input and output files are stored in the same directory as the file with the source code of the program.
