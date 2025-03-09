import numpy as np
from create_grid_from_png import grid_from_image

LEARNING_RATE = 0.9
DISCOUNT_FACTOR = 0.7
MAX_STEPS = 50000
EPSILON = 0.3

WIDTH = 25
HEIGHT = WIDTH
NUM_STATES = WIDTH*HEIGHT

np.set_printoptions(linewidth=130)
np_random = np.random.default_rng(1000)

def get_state_index(location: list):
    return location[0]+location[1]*WIDTH


ACTIONS = [tuple((x, y)) for x in range(-1, 2) for y in range(-1, 2) if not ((x==-1 and y==-1) or (x==0 and y==0))]
NUM_ACTIONS = len(ACTIONS)

Q_table = np_random.random((NUM_STATES, NUM_ACTIONS))

def q_func(state: list, action_idx: int):
    return Q_table[get_state_index(state)][action_idx]


REWARD_MATRIX = np.zeros((WIDTH, HEIGHT))

TARGET_LOCATION = [11,5]
BEGINNING_LOCATION = [23,20]

for x in range(WIDTH):
    for y in range(HEIGHT):
        REWARD_MATRIX[x][y] = (abs((TARGET_LOCATION[0]-x))+abs((TARGET_LOCATION[1]-y)))

REWARD_MATRIX = -REWARD_MATRIX
# REWARD_MATRIX -= REWARD_MATRIX.max()

# REWARD_MATRIX = -REWARD_MATRIX

# REWARD_MATRIX[TARGET_LOCATION[0]][TARGET_LOCATION[1]] = 10000

def print_coords(matrix: np.ndarray):
    print(np.flip(matrix, axis=1).T)





LAKE_GRID = grid_from_image("lake_obstacle.png")
print_coords(REWARD_MATRIX)
print_coords(LAKE_GRID)
print(ACTIONS)
# exit()


def choose_action_for_max_q(state: list) -> tuple[int, int]:
    q = []
    for idx, action in enumerate(ACTIONS):
        q.append(q_func(state=state, action_idx=idx))
    max_q = max(q)

    indices = [i for i, x in enumerate(q) if x == max_q]
    return np_random.choice(indices), max_q


def eps_greedy_policy(state: list):
    choose_from_q = np_random.choice([True, False], p=[1-EPSILON, EPSILON])

    if choose_from_q:
        return choose_action_for_max_q(state)
    else:
        random_action = np_random.choice(NUM_ACTIONS)
        return random_action, q_func(state, random_action)

    



def get_reward_and_new_state(state: list, action_idx: int) -> tuple[int, list]:
    action = ACTIONS[action_idx]
    state = state.copy()

    # check to make sure action does not move state into obstacle
    if LAKE_GRID[state[0] + action[0]][state[1]+action[1]] == 1:
        return -1000-REWARD_MATRIX.max(), state

    state[0] += action[0]
    state[1] += action[1]

    return REWARD_MATRIX[state[0]][state[1]], state






steps = 0

while True:
    location = BEGINNING_LOCATION
    if steps >= MAX_STEPS:
        break

    while True:
        # print(location)
        if steps >= MAX_STEPS:
            break

        action_idx, q = eps_greedy_policy(state=location)

        reward, new_location = get_reward_and_new_state(state=location, action_idx=action_idx)

        _, q_prime = choose_action_for_max_q(state=new_location)
        Q_table[get_state_index(location)][action_idx] = q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * q_prime - q)

        location = new_location
        steps += 1
        if (location[0] == TARGET_LOCATION[0] and location[1] == TARGET_LOCATION[1]):
            break
    print("RESET STATE")
    print(steps)

location = BEGINNING_LOCATION

for i in range(200):
    action_idx, q = choose_action_for_max_q(state=location)
    reward, new_location = get_reward_and_new_state(state=location, action_idx=action_idx)
    # print(new_location)
    LAKE_GRID[new_location[0]][new_location[1]]=8
    print(reward)
    location = new_location
    if (location[0] == TARGET_LOCATION[0] and location[1] == TARGET_LOCATION[1]):
        break

print(Q_table[get_state_index(location)])
print(ACTIONS)
print_coords(LAKE_GRID)


# Notes for presentation:
# Reward needs to be negative so it doesn't fluctuate 










        