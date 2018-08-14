from maze_dqn.maze_env import Maze
from maze_dqn.RL_brain import DeepQNetwork


def run_maze():
    step = 0
    for episode in range(300):
        observation = env.reset()
        print('\nin episode %d\n'%episode)

        while True:
            env.render()

            action = RL.choose_action(observation)

            observation_, reward, done = env.step(action)

            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                RL.learn()

            observation = observation_

            if done:
                break
            step += 1


if __name__ == '__main__':
    env = Maze()
    RL = DeepQNetwork(
        n_states=2,
        n_actions=env.n_actions,
        output_graph=True
    )
    run_maze()