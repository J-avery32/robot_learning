import time
import numpy as np
from create_grid_from_png import grid_from_image

# Got these functions from here:
# https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.round(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)), 5)




def print_coords(matrix: np.ndarray):
    print(np.flip(matrix, axis=1).T)


np.set_printoptions(linewidth=130)
np_random = np.random.default_rng()

np_sim_random = np.random.default_rng()

# Map and simulation parameters
WIDTH = 25
HEIGHT = WIDTH
NUM_STATES = WIDTH*HEIGHT
SIM_NOISE = 0.05
POLAR_DIAGRAM_VELOCITIES = {
 angle_between((1,0),(0,1)): 3.5,
 angle_between((1,0),(1,0)): 2.25,
 angle_between((-1,1),(1,0)): 3,
 angle_between((1,1),(1,0)): 3,
}

print(POLAR_DIAGRAM_VELOCITIES)

# print(POLAR_DIAGRAM_VELOCITIES)
WIND_DIRECTION = [-1,-1]
ACTIONS = [tuple((x, y)) for x in range(-1, 2) for y in range(-1, 2) if not ((x==-WIND_DIRECTION[0] and y==-WIND_DIRECTION[1]) or (x==0 and y==0))]
# print(ACTIONS)
NUM_ACTIONS = len(ACTIONS)
LAKE_GRID = grid_from_image("lake_obstacle.png")
print_coords(LAKE_GRID)

# Parameters for bonus q-learning algorithm from class
H = 5000
EPISODES = 60
confidence_parameter = 0.1
l = np.log2(NUM_ACTIONS*NUM_STATES*EPISODES/confidence_parameter)

# I am not adding a separate dimension for time while the original algorithm does
N_table_h = np.full((NUM_STATES, NUM_ACTIONS, H), 0)
V_table_h = np.full((NUM_STATES, H+1), 0)



# Parameters for Sutton and Barto algorithm
LEARNING_RATE = 0.9
DISCOUNT_FACTOR = 0.95
EPSILON = 0.3
MAX_STEPS = 20000



t0 = time.time()
# Set up data structures for learning
Q_table = np_random.random((NUM_STATES, NUM_ACTIONS))
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


def get_state_index(location: list):
    return location[0]+location[1]*WIDTH

def q_func(state: list, action_idx: int):
    return Q_table[get_state_index(state)][action_idx]

def n_h_func(state: list, action_idx: int, h: int):
    return N_table_h[get_state_index(state)][action_idx][h]

def v_h_func(state: list, h: int):
    return V_table_h[get_state_index(state)][h]




# print_coords(REWARD_MATRIX)
# print_coords(LAKE_GRID)
# print_coords(Q_table)
# print(ACTIONS)
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





def get_action(action_idx: int, noise=0) -> tuple:
    # noise must be a probability
    assert (noise < 1 and noise >= 0)

    attempted_action = ACTIONS[action_idx]

    bumped_action1 = None
    bumped_action2 = None
    if(abs(attempted_action[0])==1 and abs(attempted_action[1])==1):
        bumped_action1 = (0, attempted_action[1])
        bumped_action2 = (attempted_action[0], 0)
    elif abs(attempted_action[0]) == 1:
        bumped_action1 = (attempted_action[0], 1)
        bumped_action2 = (attempted_action[0], -1)
    else:
        bumped_action1 = (1, attempted_action[1])
        bumped_action2 = (-1, attempted_action[1])
    action = np_sim_random.choice([bumped_action1, attempted_action, bumped_action2], p=[noise/2, 1-noise, noise/2])
    return action
    
# Use dense reward function
# original algorithm from paper if noise=0
def get_reward_and_new_state_dense(state: list, action_idx: int, noise=0):
    action = get_action(action_idx, noise=noise)

    state = state.copy()

    if LAKE_GRID[state[0]][state[1]] == 1:
        return -1000-REWARD_MATRIX.max(), state

    # check to make sure action does not move state into obstacle
    if LAKE_GRID[state[0] + action[0]][state[1]+action[1]] == 1:
        return -1000-REWARD_MATRIX.max(), state
    
    state[0] += action[0]
    state[1] += action[1]

    return REWARD_MATRIX[state[0]][state[1]], state









# New algorithm using the polar diagram to give a reward for velocity
def get_reward_and_new_state_with_velocity(state: list, action_idx: int, noise=0) -> tuple[int, list]:
    
    action = get_action(action_idx, noise)
    state = state.copy()

    if LAKE_GRID[state[0]][state[1]] == 1:
        return -1000-REWARD_MATRIX.max(), state

     # check to make sure action does not move state into obstacle
    if LAKE_GRID[state[0] + action[0]][state[1]+action[1]] == 1:
        return -1000, state
    
    angle = angle_between(action, WIND_DIRECTION)

    state[0] += action[0]
    state[1] += action[1]

   
    # the more velocity you have the smaller your cost is
    # return -200, state
    factor = 1

    # Account for longer distance in a diagonal action
    if(abs(action[0])==1 and abs(action[1])==1):
        factor = 1.41

    # give a reward for getting to the terminal state, maybe it should be 
    # penalized for velocity???
    if state[0] == TARGET_LOCATION[0] and state[1] == TARGET_LOCATION[1]:
        return 0, state


    return (-factor/(POLAR_DIAGRAM_VELOCITIES[angle])), state
    # if (action[0] == -1):
    #     return -0.6, state
    # return -0.1, state






def choose_action_for_max_q_h(state: list, h, q_table_h: np.ndarray):
    q = []
    for idx, action in enumerate(ACTIONS):
        q.append(q_table_h[get_state_index(state)][idx][h])
    max_q = max(q)

    indices = [i for i, x in enumerate(q) if x == max_q]
    return np_random.choice(indices), max_q


# Not working
def bonus_alg_from_class():
    Q_table_h = np.full((NUM_STATES, NUM_ACTIONS, H), H)
    for episode in range(0,EPISODES):
        location = BEGINNING_LOCATION.copy()
        for step in range(0,H):
            action_idx, _ = choose_action_for_max_q_h(location, step, Q_table_h)
            # print("action")
            # print(ACTIONS[action_idx])
            # print("location")
            # print(location)
            # # print(q_func(location, action_idx-7))
            # if step > 100:
            #     exit()
            # TODO: train with noise
            reward, new_location = get_reward_and_new_state_with_velocity(location, action_idx)
            print("old loc",  location)
            print("action", ACTIONS[action_idx])
            print("new loc", new_location)
            # print(reward)
            N_table_h[get_state_index(location)][action_idx][step] += 1

            bonus = np.sqrt(H*H*H*l/n_h_func(location, action_idx, step))
            learn_rate = ((H+1)/(H+n_h_func(location, action_idx, step)))
            

            Q_table_h[get_state_index(location)][action_idx][step] = \
                (1-learn_rate)*Q_table_h[get_state_index(location)][action_idx][step] + \
                learn_rate*(reward + v_h_func(new_location, step+1) + bonus)
            print("HERRERERERERERERE")
            # print(bonus)
            # print(
            #     learn_rate*(reward + v_func(new_location) + bonus))
            print((reward + v_h_func(new_location, step+1) + bonus))
            _, max_q = choose_action_for_max_q_h(location, step, Q_table_h)
            V_table_h[get_state_index(location)][step] = min(H, max_q)

            location = new_location

            # Is this following the algorithm in class correctly? I don't know
            if (location[0] == TARGET_LOCATION[0] and location[1] == TARGET_LOCATION[1]):
                print("DONE")
                exit()
                break
                
            







def sutton_and_barto_alg():
    steps = 0

    while True:
        location = BEGINNING_LOCATION.copy()
        if steps >= MAX_STEPS:
            break

        while True:
            # print(location)
            if steps >= MAX_STEPS:
                break

            action_idx, q = eps_greedy_policy(state=location)

            reward, new_location = get_reward_and_new_state_dense(state=location, action_idx=action_idx, noise=0.05)

            _, q_prime = choose_action_for_max_q(state=new_location)
            Q_table[get_state_index(location)][action_idx] = q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * q_prime - q)

            location = new_location
            steps += 1
            if (location[0] == TARGET_LOCATION[0] and location[1] == TARGET_LOCATION[1]):
                # print("TARGET LOCATION")
                break
            
        # print(steps)


# used for update step of value iteration
def max_value_over_actions(V_table: np.ndarray, state: list, velocity=False) -> tuple[int, int]:
    all_value_action_pairs = []
    for idx in range(NUM_ACTIONS):
        if velocity:
            reward, new_state = get_reward_and_new_state_with_velocity(state=state,action_idx=idx)
        else:
            reward, new_state = get_reward_and_new_state_dense(state=state, action_idx=idx)
        all_value_action_pairs.append((reward + DISCOUNT_FACTOR * V_table[get_state_index(new_state)], idx))
    return max(all_value_action_pairs, key= lambda k: k[0])



def value_iteration(velocity=False):
    theta = 0.01
    V_table = np_random.random(size=NUM_STATES)
    V_table[get_state_index(TARGET_LOCATION)] = 0

    while (True):
        delta = 0
        for x in range(WIDTH):
            for y in range(HEIGHT):
                v = V_table[get_state_index([x, y])]
                V_table[get_state_index([x, y])] = max_value_over_actions(V_table, [x, y], velocity=velocity)[0]
                delta = max(delta, abs(v-V_table[get_state_index([x, y])]))
        if (delta < theta):
            break
    return V_table


# sutton_and_barto_alg()



def policy_from_value_iteration(velocity=False):
    print("Value iteration")
    V_table =value_iteration(velocity=velocity)
    print("TAREGT VALUE", V_table[get_state_index(TARGET_LOCATION)])
    location = BEGINNING_LOCATION
    total_reward = 0
    for _ in range(100):
        _, action_idx = max_value_over_actions(V_table=V_table, state=location, velocity=velocity)
        print("VALUE:", _)
        reward, new_location = get_reward_and_new_state_with_velocity(state=location, action_idx=action_idx)

        total_reward += reward
        print(reward)
        print(new_location)
        # print(new_location)
        LAKE_GRID[new_location[0]][new_location[1]]=8
        # print(reward)
        location = new_location
        if (location[0] == TARGET_LOCATION[0] and location[1] == TARGET_LOCATION[1]):
            action = ACTIONS[action_idx]
            angle = angle_between(action, WIND_DIRECTION)
            factor = 1

            if(abs(action[0])==1 and abs(action[1])==1):
                factor = 1.41
            total_reward += -factor/POLAR_DIAGRAM_VELOCITIES[angle] 
            

            print("TERMINAL STATE")
            print("Time:", total_reward)
            break




def policy_from_q_learning():
    print("Q-LEARNING")
    sutton_and_barto_alg()
    location = BEGINNING_LOCATION
    total_reward = 0
    for i in range(100):
        action_idx, q = choose_action_for_max_q(state=location)
        reward, new_location = get_reward_and_new_state_with_velocity(state=location, action_idx=action_idx)
        total_reward += reward
        # print(new_location)
        LAKE_GRID[new_location[0]][new_location[1]]=8
        # print(reward)
        location = new_location
        if (location[0] == TARGET_LOCATION[0] and location[1] == TARGET_LOCATION[1]):
            print("TIME:", total_reward)
            break
        



# policy_from_q_learning()

policy_from_value_iteration(velocity=True)

print (time.time() - t0), "seconds process time"
# print(Q_table[(location)])
# print(ACTIONS)
print_coords(LAKE_GRID)


# Notes for presentation:
# Reward needs to be negative so it doesn't fluctuate 










        