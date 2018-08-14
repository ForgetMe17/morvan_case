import pandas as pd
import numpy as np
import time

N_STATES = 6
ACTIONS = ['left','right']
EPSILON = 0.9    # greedy policy
ALPHA = 0.1     # learning rate
LAMBDA = 0.9   # discount factor
MAX_EPISODES = 13
FRESH_TIME = 0.3

np.random.seed(2)


def build_q_table(n_state, actions):
    q_table = pd.DataFrame(
        np.zeros((n_state, len(actions))),
        columns=actions
    )
    return q_table


def choose_action(state, q_table):
    q_state = q_table.iloc[state,:]
    if (np.random.uniform()>EPSILON) or ((q_state==0).all()):
        action = np.random.choice(ACTIONS)
    else:
        action = q_state.idxmax()
    return action


def get_env_feedback(s, a):
    if a == 'left':
        r = 0
        if s == 0:
            s_ = s
        else:
            s_ = s-1
    else:
        if s == N_STATES-2:
            r = 1
            s_ = 'T'
        else:
            r = 0
            s_ = s+1
    return s_, r


def update_env(s, episode, step_count):
    if s == 'T':
        interaction = 'Episode %s: total steps %s' % (episode+1, step_count+1)
        print('\r{}'.format(interaction), end='')
        time.sleep(1)
        print('\r      ', end='')
    else:
        env_list = ['-']*(N_STATES-1)+['T']
        env_list[s] = '0'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


def rl():
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(N_STATES):
        s = 0
        step_count = 0
        is_terminate = False
        update_env(s, episode, step_count)
        while not is_terminate:
            a = choose_action(s, q_table)
            q_predict = q_table.loc[s, a]
            s_, r = get_env_feedback(s, a)
            if s_ == 'T':
                is_terminate = True
                q_target = r
            else:
                q_target = r + LAMBDA*q_table.iloc[s_, :].max()
            q_table.loc[s, a] += ALPHA*(q_target-q_predict)
            s = s_
            update_env(s, episode, step_count)
            step_count += 1
    return  q_table


if __name__ == '__main__':
    q_table = rl()
    print(q_table)