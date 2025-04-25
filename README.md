# Schedule Optimizer 🗓️ ⚡

An intelligent scheduling system that optimizes timetables using dynamic programming with bitmask optimization. Maximizes lesson variety while considering time constraints and travel distances between locations.

## Purpose 🎯
The program `schedule_bitmask_dp.py` generates optimal schedules through mathematical optimization, prioritizing:
1. **Maximum Diversity**: Largest possible number of different lesson types (user-selectable priority)
2. **Weekend Adjacency**: Tie-breaker: Most consecutive free days near weekends
3. **Custom Vacancy Patterns**: Optimizes for vacancies at start/end/both ends of week

## Key Features ✨

- **Optimal Schedule Generation**:
  - Configurable priority system for lesson types
  - Weekend-adjacent free day maximization
  - Three vacancy optimization modes (start/end/both)
  
- **Custom Constraints**:
  - **Mandatory Lessons**: Force inclusion of specific classes
  - **Time Windows**: Reserve periods for meals/breaks
  - **Location Awareness**: Automatic travel time calculations

- **Performance**:
  - Bitmask DP implementation (O(N²·2ᴺ) complexity)
  - Parallel computation support
  - Real-time runtime statistics

## Technical Considerations ⚠️

- **Input/Output**: 
  - Requires `schedule_input.txt` in program directory
  - Generates `schedule_output.txt` in same directory
- **Limitations**:
  - ≤24 lesson types recommended for reasonable runtime
  - Location indexes must be 0-(N-1) for N locations
- **Complexity Factors**:
  - Primary constraint: Number of lesson types
  - Secondary constraint: Number of lesson instances

## Installation & Requirements 🛠️

### Prerequisites
- Python 3.9+
- 4GB RAM (recommended)

```bash
git clone https://github.com/yourusername/schedule-optimizer.git
cd schedule-optimizer



## Input Format 📄
### Distance Matrix Structure

```bash
python
dist = [
    [0, 5, 7],  # From location 0
    [5, 0, 3],  # From location 1
    [7, 3, 0]   # From location 2
]

- Row index = source location
- Column index = destination location
- Values represent walking time in minutes


## Sample Input Configuration


``` bash
mandatory = [0, 2]  # Must include lessons with these IDs
free_windows = [("Monday", "11:30", "13:00")]
vacancy_mode = "END"  # Options: START/END/BOTH
lesson_preferences = {3: 2.0, 5: 1.5}  # Weighted priorities



## Usage & Output 🖥️

1. Configure schedule_input.txt following examples
2. Run program: python schedule_bitmask_dp.py
3. Results saved to schedule_output.txt

### Output Includes:

- Optimal lesson combination with times
- Travel time calculations between locations
- Free period analysis
- Vacancy statistics
- Runtime metrics

## Performance Notes ⏱️
### Typical Execution Times (M1 Pro processor):

- 16 lessons: ~2 seconds
- 20 lessons: ~30 seconds
- 24 lessons: ~8 minutes

### Optimization Tips:

- Group lessons by location when possible
- Use time windows to reduce solution space
- Start with mandatory lessons first
- Experiment using runtime statistics for local tuning
