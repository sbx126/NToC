import numpy as np
import sys
from copy import deepcopy


class Hanoi():
    def __init__(self):
        self.num_disks = 3
        self.pillars = [[] for _ in range(3)]

        self.last_action = None

    def reset(self, num_disks = None):
        if num_disks is not None:
            self.num_disks = num_disks
        assert self.num_disks >= 1
        
        for pillar in self.pillars:
            pillar.clear()
        
        for i in range(self.num_disks-1, -1, -1):
            self.pillars[0].append(i)

        self.last_action = None

        return deepcopy(self.pillars)

    def step(self, action):
        assert action[0] != action[1]

        if self._pillar_len(action[0]) > 0 and self._pillar_size(action[0]) < self._pillar_size(action[1]):
            disk = self.pillars[action[0]].pop()
            self.pillars[action[1]].append(disk)

        self.last_action = deepcopy(action)

        return deepcopy(self.pillars), self._env_done()

    def optimal_policy(self):
        if self.last_action is None:
            if self.num_disks % 2 == 1:
                action = (0, 2)
            else:
                action = (0, 1)
        elif self._smallest_pillar() == self.last_action[1]:
            # move the second-smallest disk
            action = (self._second_smallest_pillar(), self._largest_pillar())
        else:
            if self.num_disks % 2 == 1:
                # move the smallest disk left
                action = (self._smallest_pillar(), (self._smallest_pillar() - 1) % 3)
            else:
                # move the smallest disk right
                action = (self._smallest_pillar(), (self._smallest_pillar() + 1) % 3)

        return action

    def record_traj(self, num_disks):
        traj, acts = [], []
        state = self.reset(num_disks = num_disks)
        traj.append(deepcopy(state))
        done = False
        while not done:
            action = self.optimal_policy()
            state, done = self.step(action)
            
            traj.append(deepcopy(state))
            acts.append(deepcopy(action))

        return traj, acts

    def render(self):
        for i in range(self.num_disks):
            a = self.pillars[0][self.num_disks-1-i] if (len(self.pillars[0]) >= self.num_disks - i) else " "
            b = self.pillars[1][self.num_disks-1-i] if (len(self.pillars[1]) >= self.num_disks - i) else " "
            c = self.pillars[2][self.num_disks-1-i] if (len(self.pillars[2]) >= self.num_disks - i) else " "
            print("|{}|{}|{}|".format(a, b, c))
        print("=======")

    def _pillar_size(self, idx):
        if len(self.pillars[idx]) == 0:
            return 100000
        else:
            return self.pillars[idx][-1]

    def _pillar_len(self, idx):
        return len(self.pillars[idx])

    def _env_done(self):
        if self._pillar_len(0) == 0 and self._pillar_len(1) == 0:
            return True
        else:
            return False

    def _smallest_pillar(self):
        if self._pillar_size(0) < self._pillar_size(1):
            idx = 0
        else:
            idx = 1

        if self._pillar_size(2) < self._pillar_size(idx):
            idx = 2

        return idx

    def _second_smallest_pillar(self):
        if self._pillar_size(0) < self._pillar_size(1):
            if self._pillar_size(2) < self._pillar_size(0):
                return 0
            elif self._pillar_size(1) < self._pillar_size(2):
                return 1
            else:
                return 2
        else:
            if self._pillar_size(1) > self._pillar_size(2):
                return 1
            elif self._pillar_size(2) > self._pillar_size(0):
                return 0
            else:
                return 2

    def _largest_pillar(self):
        if self._pillar_size(0) > self._pillar_size(1):
            idx = 0
        else:
            idx = 1

        if self._pillar_size(2) > self._pillar_size(idx):
            idx = 2

        return idx


if __name__ == "__main__":
    env = Hanoi()

    state = env.reset(num_disks = 5)
    env.render()
    done = False
    c = 0
    while not done:
        action = env.optimal_policy()
        state, done = env.step(action)
        print(action)
        env.render()

        c += 1
        if c >= 50:
            break


