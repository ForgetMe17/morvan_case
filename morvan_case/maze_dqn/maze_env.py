import tkinter as tk
import numpy as np
import time

UNIT = 80
SIZE = UNIT/2 - 5
MAZE_H = 4
MAZE_W = 4


class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.title('Maze')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_W * UNIT))
        self._build_maze()

    def _build_maze(self):
        # canvas
        self.canvas = tk.Canvas(self, bg='white',
                                height=MAZE_H * UNIT,
                                width=MAZE_W * UNIT)
        # grids
        for c in range (0, UNIT * MAZE_W, UNIT):
            x0, y0, x1, y1 = c, 0, c, UNIT * MAZE_H
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, UNIT * MAZE_H, UNIT):
            x0, y0, x1, y1 = 0, r, UNIT * MAZE_W, r
            self.canvas.create_line(x0, y0, x1, y1)

        # origin
        origin = np.array([UNIT/2, UNIT/2])

        # hell
        hell1_center = origin + np.array([1 * UNIT, 2 * UNIT])
        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0]-SIZE, hell1_center[1]-SIZE,
            hell1_center[0]+SIZE, hell1_center[1]+SIZE,
            fill='black'
        )

        hell2_center = origin + np.array([2 * UNIT, 1 * UNIT])
        self.hell2 = self.canvas.create_rectangle(
            hell2_center[0]-SIZE, hell2_center[1]-SIZE,
            hell2_center[0]+SIZE, hell2_center[1]+SIZE,
            fill='black'
        )

        # oval
        oval_center = origin + np.array([2 * UNIT, 2 * UNIT])
        self.oval = self.canvas.create_rectangle(
            oval_center[0] - SIZE, oval_center[1] - SIZE,
            oval_center[0] + SIZE, oval_center[1] + SIZE,
            fill='yellow'
        )

        # red rect
        self.rect = self.canvas.create_rectangle(
            origin[0] - SIZE, origin[1] - SIZE,
            origin[0] + SIZE, origin[1] + SIZE,
            fill='red')

        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.rect)
        origin = np.array([UNIT/2, UNIT/2])
        self.rect = self.canvas.create_rectangle(
            origin[0] - SIZE, origin[1] - SIZE,
            origin[0] + SIZE, origin[1] + SIZE,
            fill='red')
        return (np.array(self.canvas.coords(self.rect)[:2]) -
                np.array(self.canvas.coords(self.oval)[:2])) / (MAZE_H*UNIT)

    def step(self, action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0,0])
        if action == 0:  # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:  # down
            if s[1] < UNIT * (MAZE_H-1):
                base_action[1] += UNIT
        elif action == 2:  # right
            if s[0] < (MAZE_W-1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:  # left
            if s[0] > UNIT:
                base_action[0] -= UNIT

        self.canvas.move(self.rect, base_action[0],base_action[1])

        s_ = self.canvas.coords(self.rect)

        if s_ == self.canvas.coords(self.oval):
            r = 1
            done = True
        elif s_ in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2)]:
            r = -1
            done = True
        else:
            r = 0
            done = False
        s_ = (np.array(s_[:2]) - np.array(self.canvas.coords(self.oval)[:2])) / (MAZE_H*UNIT)
        return s_, r, done

    def render(self):
        time.sleep(0.01)
        self.update()


def cor_trans(ob_lsit):
    x = int(ob_lsit[0]/UNIT)
    y = int(ob_lsit[1]/UNIT)
    return [x, y]


def update():
    for t in range(10):
        s = env.reset()
        while True:
            env.render()
            a = 2
            s, r, done = env.step(a)
            if done:
                break


if __name__ == '__main__':
    env = Maze()
    env.after(100, update)
    env.mainloop()