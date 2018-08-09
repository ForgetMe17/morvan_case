from brain import QLearningTable
from maze_env import Maze


def update():
    for episode in range(100):
        observation = env.reset()
        while True:
            env.render()

            action = RL.choose_action(str(observation))

            observation_, r, done = env.step(action)

            RL.learn(str(observation), action, r, str(observation_))

            observation = observation_

            if done:
                print('Episido %s : q_table' % episode)
                print(RL.q_table.sort_index())
                break


if __name__ == "__main__":
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()